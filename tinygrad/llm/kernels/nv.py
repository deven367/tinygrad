from __future__ import annotations
import functools, math
from typing import Any
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

Q8_0, GGML_BLOCK_SIZE, Q8_GROUP_SIZE, Q8_U16_WORDS = 8, 32, 32, 17


def nv_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if isinstance(device, tuple): device = device[0]
  return device is not None and device.split(":")[0] in ("NV", "CUDA")


def _nv_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.int32, (a.cast(dtypes.uint32), b.cast(dtypes.uint32), c), arg="__dp4a((int){0}, (int){1}, {2})")


def _nv_shuffle_xor(value:UOp, offset:int) -> UOp:
  return UOp(Ops.CUSTOMI, value.dtype, (value,), arg=f"__shfl_xor_sync(0xffffffff, {{0}}, {offset}, 32)")


def _nv_shuffle_idx(value:UOp, idx:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, value.dtype, (value, idx.cast(dtypes.int32)), arg="__shfl_sync(0xffffffff, {0}, {1}, 32)")


def _nv_fmax(a:UOp, b:UOp) -> UOp:
  # NOTE: cannot use UOp.maximum: the CUDA renderer lowers it to a ternary
  # (a < shfl) ? shfl : a, evaluating the shuffle twice. Double-evaluated
  # __shfl_xor_sync returns garbage (observed alternating 0/32 across lanes).
  return UOp(Ops.CUSTOMI, dtypes.float32, (a, b), arg="fmaxf({0}, {1})")

def _nv_ldcs16(ptr:UOp) -> UOp:
  # aligned 2-byte streaming load: weights are read once per token, skip L2 allocation
  return UOp(Ops.CUSTOMI, dtypes.uint16, (ptr,), arg="__ldcs((const unsigned short*){0})")


def _warp_reduce(value:UOp, maximum:bool=False) -> UOp:
  for offset in (16, 8, 4, 2, 1):
    other = _nv_shuffle_xor(value, offset)
    value = _nv_fmax(value, other) if maximum else value + other
  return value


def _load_lanes(ptr:UOp, lanes:int) -> UOp:
  assert ptr.op is Ops.INDEX
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(dtype=ptr.dtype)


def _half(bits:UOp) -> UOp: return bits.cast(dtypes.uint16).bitcast(dtypes.float16).float()


@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features // Q8_GROUP_SIZE
  token_group = UOp.range(tokens*groups, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token, group = token_group//groups, token_group%groups
  x = x.reshape(tokens, groups, Q8_GROUP_SIZE)
  v = x[token, group, lane].float()
  group_scale = (_warp_reduce(v.abs(), maximum=True) / 127).maximum(1e-8)
  q_val = (v / group_scale).round().clip(-127, 127).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32)
  word_lane = lane.minimum(7)
  q0 = _nv_shuffle_idx(q_val, word_lane * 4)
  q1 = _nv_shuffle_idx(q_val, word_lane * 4 + 1)
  q2 = _nv_shuffle_idx(q_val, word_lane * 4 + 2)
  q3 = _nv_shuffle_idx(q_val, word_lane * 4 + 3)
  word = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
  # q holds 8 words (32 B) per group: lanes 0..7 store the valid words, lanes 8..31
  # duplicate word 7 (same value, keeps all 32 lanes active — no warp predication).
  # scale holds 1 f32 per group: all lanes store the identical value (one 32 B sector).
  stores = (q[token, group, word_lane].store(word), scale[token, group, 0].store(group_scale))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="nv_q8_quantize", opts_to_apply=()))


def _q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  if not hasattr(x, '_q8_cache'): x._q8_cache = {}
  key = (tokens, in_features)
  if key in x._q8_cache: return x._q8_cache[key]
  groups = in_features // Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, 1, dtype=dtypes.float32, device=x.device)
  q, scale = Tensor.custom_kernel(q, scale, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:2]
  x._q8_cache[key] = (q, scale)
  return q, scale




@functools.cache
def _rmsnorm_kernel(out:UOp, x:UOp, weight:UOp, tokens:int, dim:int, eps:float) -> UOp:
  token = UOp.range(tokens, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  elems = dim // 32
  acc = UOp.const(0, dtypes.float32)
  for i in range(elems):
    idx = lane + i * 32
    v = x[token, idx].float()
    acc = acc + v * v
  total = _warp_reduce(acc)
  norm = (total / dim + eps).rsqrt()
  stores = []
  for i in range(elems):
    idx = lane + i * 32
    v = x[token, idx].float()
    stores.append(out[token, idx].store((v * norm * weight[idx].float()).cast(out.dtype)))
  return UOp.group(*stores).end(token, lane).sink(arg=KernelInfo(name="nv_rmsnorm", opts_to_apply=()))


def nv_rmsnorm(x:Tensor, weight:Tensor, eps:float=1e-6) -> Tensor:
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or isinstance(D, UOp) or D % 32 != 0:
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight
  orig_shape = x.shape
  x_flat = x.reshape(-1, D)
  tokens = x_flat.shape[0]
  out = Tensor.empty(tokens, D, dtype=x.dtype, device=x.device)
  res = Tensor.custom_kernel(out, x_flat.contiguous(), weight.contiguous(),
                             fxn=functools.partial(_rmsnorm_kernel, tokens=tokens, dim=D, eps=eps))[0]
  return res.reshape(orig_shape)


@functools.cache
def _add_rmsnorm_kernel(h_out:UOp, norm_out:UOp, x:UOp, res:UOp, weight:UOp, tokens:int, dim:int, eps:float) -> UOp:
  # Fused residual add + RMSNorm: h = x + res, out = rmsnorm(h) * weight.
  # The per-lane sums are single UOp values reused in both the accumulation and the
  # stores, so the renderer keeps them in registers (no DRAM round-trip for h before
  # normalization). The sum keeps the source dtype, matching the standalone add kernel
  # bit-for-bit (the E_* add of the unfused path).
  # ponytail: dims up to 5120 hold 160 f32 sums/thread (~200 regs); if a wider dim
  # spills, reload x/res in the store pass (like _rmsnorm_kernel) — costs 2 loads/elt.
  token = UOp.range(tokens, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  elems = dim // 32
  vals = []
  acc = UOp.const(0, dtypes.float32)
  for i in range(elems):
    idx = lane + i * 32
    v = x[token, idx] + res[token, idx]
    vals.append(v)
    acc = acc + v.float() * v.float()
  total = _warp_reduce(acc)
  norm = (total / dim + eps).rsqrt()
  stores = []
  for i, v in enumerate(vals):
    idx = lane + i * 32
    stores.append(h_out[token, idx].store(v))
    stores.append(norm_out[token, idx].store((v.float() * norm * weight[idx].float()).cast(norm_out.dtype)))
  return UOp.group(*stores).end(token, lane).sink(arg=KernelInfo(name="nv_add_rmsnorm", opts_to_apply=()))


def nv_add_rmsnorm(x:Tensor, residual:Tensor, weight:Tensor, eps:float=1e-6) -> tuple[Tensor, Tensor]:
  """Fused residual add + RMSNorm: returns (h, rmsnorm(h) * weight) with h = x + residual."""
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or weight is None or isinstance(D, UOp) or D % 32 != 0:
    h = x + residual
    return h, nv_rmsnorm(h, weight, eps)
  orig_shape = x.shape
  x_flat = x.reshape(-1, D)
  r_flat = residual.reshape(-1, D)
  tokens = x_flat.shape[0]
  h = Tensor.empty(tokens, D, dtype=x.dtype, device=x.device)
  out = Tensor.empty(tokens, D, dtype=x.dtype, device=x.device)
  h, out = Tensor.custom_kernel(h, out, x_flat.contiguous(), r_flat.contiguous(), weight.contiguous(),
                                fxn=functools.partial(_add_rmsnorm_kernel, tokens=tokens, dim=D, eps=eps))[:2]
  return h.reshape(orig_shape), out.reshape(orig_shape)


def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str="nv_linear_q8_0") -> UOp:
  chunks = (group_count+31)//32
  token_output = UOp.range(out.shape[0]*out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token = token_output // out_features
  output = token_output % out_features
  acc = UOp.const(0, dtypes.float32)
  for chunk in range(chunks):
    group = lane + chunk * 32
    val = group_dot(token, output, group) if group_count % 32 == 0 else       (group < group_count).where(group_dot(token, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))
    acc = acc + val
  total = _warp_reduce(acc)
  return out[token, output, lane].store(total.cast(out.dtype)).end(token_output, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))


@functools.cache
def _q8_0_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    xwords = _load_lanes(xq[token, group, 0], 8)
    base = (output*group_count+group)*Q8_U16_WORDS
    dot = UOp.const(0, dtypes.int32)
    for word_idx in range(8):
      word = _nv_ldcs16(raw[base+1+word_idx*2]) | (_nv_ldcs16(raw[base+2+word_idx*2]).cast(dtypes.uint32) << 16)
      dot = _nv_dp4a(word, xwords[word_idx], dot)
    return dot.float() * xd[token, group, 0] * _half(_nv_ldcs16(raw[base]))
  return _decode_linear(out, out_features, group_count, group_dot)


def q8_0_linear(layer:Any, x:Tensor) -> Tensor:
  assert layer.ggml_type == Q8_0 and layer.in_features % Q8_GROUP_SIZE == 0
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  xq, xd = _q8_quantize(x, tokens, in_features)
  out = Tensor.empty(tokens, out_features, 32, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xq.uop, xd.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
  kernel = _q8_0_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))[..., 0]
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias


@functools.cache
def _l2norm_kernel(out:UOp, x:UOp, tokens:int, dim:int, eps:float) -> UOp:
  token = UOp.range(tokens, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  elems = dim // 32
  acc = UOp.const(0, dtypes.float32)
  for i in range(elems):
    idx = lane + i * 32
    v = x[token, idx].float()
    acc = acc + v * v
  total = _warp_reduce(acc)
  # L2 norm: x / max(sqrt(sum(x²)), eps)  =>  x * (1 / max(sqrt(total), eps))
  norm = total.sqrt().maximum(eps).reciprocal()
  stores = []
  for i in range(elems):
    idx = lane + i * 32
    v = x[token, idx].float()
    stores.append(out[token, idx].store((v * norm).cast(out.dtype)))
  return UOp.group(*stores).end(token, lane).sink(arg=KernelInfo(name="nv_normalize", opts_to_apply=()))


def nv_normalize(x:Tensor, eps:float=1e-6) -> Tensor:
  """Fused warp-level L2 normalization: x / max(||x||_2, eps)."""
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or isinstance(D, UOp) or D % 32 != 0:
    return x.normalize(dim=-1, eps=eps)
  orig_shape = x.shape
  x_flat = x.reshape(-1, D)
  tokens = x_flat.shape[0]
  out = Tensor.empty(tokens, D, dtype=x.dtype, device=x.device)
  res = Tensor.custom_kernel(out, x_flat.contiguous(),
                             fxn=functools.partial(_l2norm_kernel, tokens=tokens, dim=D, eps=eps))[0]
  return res.reshape(orig_shape)
