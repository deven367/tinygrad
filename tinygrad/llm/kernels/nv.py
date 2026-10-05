from __future__ import annotations
import functools, math
from typing import Any
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops

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
  out = Tensor.empty(tokens, D, dtype=dtypes.half, device=x.device)
  h, out = Tensor.custom_kernel(h, out, x_flat.contiguous(), r_flat.contiguous(), weight.contiguous(),
                                fxn=functools.partial(_add_rmsnorm_kernel, tokens=tokens, dim=D, eps=eps))[:2]
  return h.reshape(orig_shape), out.reshape(orig_shape)


@functools.cache
def _gated_rmsnorm_kernel(out:UOp, x:UOp, gate:UOp, weight:UOp, tokens:int, dim:int, eps:float, is_kda:bool=False) -> UOp:
  token = UOp.range(tokens, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  elems = dim // 32
  vals = []
  acc = UOp.const(0, dtypes.float32)
  for i in range(elems):
    idx = lane + i * 32
    v = x[token, idx].float()
    vals.append(v)
    acc = acc + v * v
  total = _warp_reduce(acc)
  norm = (total / dim + eps).rsqrt()
  stores = []
  for i, v in enumerate(vals):
    idx = lane + i * 32
    normed = v * norm * weight[idx].float()
    g = gate[token, idx].float()
    act = g.sigmoid() if is_kda else g.silu()
    stores.append(out[token, idx].store((normed * act).cast(out.dtype)))
  return UOp.group(*stores).end(token, lane).sink(arg=KernelInfo(name="nv_gated_rmsnorm", opts_to_apply=()))


def nv_gated_rmsnorm(x:Tensor, gate:Tensor, weight:Tensor, eps:float=1e-6, is_kda:bool=False, out_dtype=dtypes.half) -> Tensor:
  """Fused RMSNorm + Gate (silu or sigmoid) projection: computes (rmsnorm(x) * weight * act(gate)).cast(out_dtype)."""
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or weight is None or isinstance(D, UOp) or D % 32 != 0:
    act = gate.sigmoid() if is_kda else gate.silu()
    return (nv_rmsnorm(x, weight, eps) * act).cast(out_dtype)
  orig_shape = x.shape
  x_flat = x.reshape(-1, D)
  g_flat = gate.reshape(-1, D)
  tokens = x_flat.shape[0]
  out = Tensor.empty(tokens, D, dtype=out_dtype, device=x.device)
  res = Tensor.custom_kernel(out, x_flat, g_flat, weight,
                             fxn=functools.partial(_gated_rmsnorm_kernel, tokens=tokens, dim=D, eps=eps, is_kda=is_kda))[0]
  return res.reshape(orig_shape)



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
  if len(out.shape) == 2:
    # Every lane must finish the collective before lane zero stores the result.
    reduced = UOp.placeholder((1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
    total = reduced.after(reduced.store(total))[0].load()
    return out[token, output.valid(lane.eq(0))].store(total.cast(out.dtype)).end(token_output, lane).sink(
      arg=KernelInfo(name=name, opts_to_apply=()))
  index = lane
  if out.shape[-1] == 1:
    # Every lane must finish the collective before lane zero stores the compact result.
    reduced = UOp.placeholder((1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
    total = reduced.after(reduced.store(total))[0].load()
    index = lane.valid(lane.eq(0))
  return out[token, output, index].store(total.cast(out.dtype)).end(token_output, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))


@functools.cache
def _q8_0_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    xwords = _load_lanes(xq[token, group, 0], 8)
    base = (output*group_count+group)*Q8_U16_WORDS
    arg_str = """([&](){{
      const unsigned short *w = (const unsigned short*)({0});
      unsigned short s1 = __ldcs(w + 1);
      unsigned short s2 = __ldcs(w + 2);
      unsigned short s3 = __ldcs(w + 3);
      unsigned short s4 = __ldcs(w + 4);
      unsigned short s5 = __ldcs(w + 5);
      unsigned short s6 = __ldcs(w + 6);
      unsigned short s7 = __ldcs(w + 7);
      unsigned short s8 = __ldcs(w + 8);
      unsigned short s9 = __ldcs(w + 9);
      unsigned short s10 = __ldcs(w + 10);
      unsigned short s11 = __ldcs(w + 11);
      unsigned short s12 = __ldcs(w + 12);
      unsigned short s13 = __ldcs(w + 13);
      unsigned short s14 = __ldcs(w + 14);
      unsigned short s15 = __ldcs(w + 15);
      unsigned short s16 = __ldcs(w + 16);

      int w0 = (int)((unsigned int)s1 | ((unsigned int)s2 << 16));
      int w1 = (int)((unsigned int)s3 | ((unsigned int)s4 << 16));
      int w2 = (int)((unsigned int)s5 | ((unsigned int)s6 << 16));
      int w3 = (int)((unsigned int)s7 | ((unsigned int)s8 << 16));
      int w4 = (int)((unsigned int)s9 | ((unsigned int)s10 << 16));
      int w5 = (int)((unsigned int)s11 | ((unsigned int)s12 << 16));
      int w6 = (int)((unsigned int)s13 | ((unsigned int)s14 << 16));
      int w7 = (int)((unsigned int)s15 | ((unsigned int)s16 << 16));

      int a0 = __dp4a(w0, (int){1}, 0);
      int a1 = __dp4a(w1, (int){2}, 0);
      int a2 = __dp4a(w2, (int){3}, 0);
      int a3 = __dp4a(w3, (int){4}, 0);
      a0 = __dp4a(w4, (int){5}, a0);
      a1 = __dp4a(w5, (int){6}, a1);
      a2 = __dp4a(w6, (int){7}, a2);
      a3 = __dp4a(w7, (int){8}, a3);

      return (a0 + a1) + (a2 + a3);
    }})()"""
    srcs = (raw[base], xwords[0], xwords[1], xwords[2], xwords[3], xwords[4], xwords[5], xwords[6], xwords[7])
    dot = UOp(Ops.CUSTOMI, dtypes.int32, srcs, arg=arg_str)
    scale = _half(_nv_ldcs16(raw[base]))
    return dot.float() * xd[token, group, 0] * scale

  return _decode_linear(out, out_features, group_count, group_dot)


def q8_0_linear(layer:Any, x:Tensor) -> Tensor:
  assert layer.ggml_type == Q8_0 and layer.in_features % Q8_GROUP_SIZE == 0
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  xq, xd = _q8_quantize(x, tokens, in_features)
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xq.uop, xd.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
  kernel = _q8_0_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias


@functools.cache
def _l2norm_kernel(out:UOp, x:UOp, tokens:int, dim:int, eps:float, scale:float=1.0, repeat:int=1) -> UOp:
  token = UOp.range(tokens, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  elems = dim // 32
  vals = []
  acc = UOp.const(0, dtypes.float32)
  for i in range(elems):
    idx = lane + i * 32
    v = x[token, idx].float()
    vals.append(v)
    acc = acc + v * v
  total = _warp_reduce(acc)
  norm = total.sqrt().maximum(eps).reciprocal()
  if scale != 1.0: norm = norm * UOp.const(scale, dtypes.float32)
  stores = []
  for i, v in enumerate(vals):
    idx = lane + i * 32
    res_val = (v * norm).cast(out.dtype)
    if repeat == 1:
      stores.append(out[token, idx].store(res_val))
    else:
      for r in range(repeat):
        stores.append(out[r, token, idx].store(res_val))
  return UOp.group(*stores).end(token, lane).sink(arg=KernelInfo(name="nv_normalize", opts_to_apply=()))


def nv_normalize(x:Tensor, eps:float=1e-6, scale:float=1.0, repeat:int=1) -> Tensor:
  """Fused warp-level L2 normalization: x / max(||x||_2, eps) * scale with optional head repeat."""
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or isinstance(D, UOp) or D % 32 != 0:
    res = x.normalize(dim=-1, eps=eps)
    if scale != 1.0: res = res * scale
    return res if repeat == 1 else res.repeat(*((1,)*(len(x.shape)-2)), repeat, 1)
  orig_shape = (*x.shape[:-2], x.shape[-2] * repeat, D) if repeat > 1 else x.shape
  x_flat = x.reshape(-1, D)
  tokens = x_flat.shape[0]
  out_shape = (repeat, tokens, D) if repeat > 1 else (tokens, D)
  out = Tensor.empty(*out_shape, dtype=x.dtype, device=x.device)
  res = Tensor.custom_kernel(out, x_flat,
                             fxn=functools.partial(_l2norm_kernel, tokens=tokens, dim=D, eps=eps, scale=scale, repeat=repeat))[0]
  return res.reshape(orig_shape)


# **************** Fused 2-Stage Argmax Reduction ****************

def _nv_pack64(f:UOp, idx:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.uint64, (f, idx),
             arg="([&](float f, int idx) {{ unsigned int u = __float_as_uint(f); unsigned int s = u ^ (((int)u >> 31) | 0x80000000); return (((unsigned long long)s << 32) | (unsigned int)(~idx)); }}({0}, {1}))")

def _nv_unpack_idx(packed:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.int32, (packed,),
             arg="((int)(~(unsigned int)({0})))")

def _nv_umax64(a:UOp, b:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.uint64, (a, b),
             arg="max((unsigned long long){0}, (unsigned long long){1})")

def _nv_shuffle_xor64(val:UOp, offset:int) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.uint64, (val,),
             arg=f"__shfl_xor_sync(0xffffffff, (unsigned long long){{0}}, {offset}, 32)")

@functools.cache
def _argmax_partial_kernel(pout:UOp, x:UOp, tokens:int, num_blocks:int, tile:int) -> UOp:
  elems = tile * 32
  token_blk = UOp.range(tokens * num_blocks, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token = token_blk // num_blocks
  blk = token_blk % num_blocks

  best = UOp.const(0, dtypes.uint64)
  for step in range(tile):
    flat_lane_idx = blk * elems + step * 32 + lane
    v = x[token, flat_lane_idx].load().float()
    step_idx = blk * elems + step * 32 + lane
    packed = _nv_pack64(v, step_idx)
    best = _nv_umax64(best, packed)

  for offset in (16, 8, 4, 2, 1):
    other = _nv_shuffle_xor64(best, offset)
    best = _nv_umax64(best, other)

  store = pout[token, blk, lane].store(best)
  return store.end(token_blk, lane).sink(arg=KernelInfo(name="nv_argmax_partial", opts_to_apply=()))

@functools.cache
def _argmax_final_kernel(out:UOp, pin:UOp, tokens:int, num_blocks:int) -> UOp:
  token = UOp.range(tokens, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)

  iters = (num_blocks + 31) // 32
  best = UOp.const(0, dtypes.uint64)
  for i in range(iters):
    b = lane + i * 32
    valid = b < num_blocks
    p = valid.where(pin[token, b.minimum(num_blocks - 1), lane].load(), UOp.const(0, dtypes.uint64))
    best = _nv_umax64(best, p)

  for offset in (16, 8, 4, 2, 1):
    other = _nv_shuffle_xor64(best, offset)
    best = _nv_umax64(best, other)

  final_idx = _nv_unpack_idx(best)
  store = out[token, lane].store(final_idx)
  return store.end(token, lane).sink(arg=KernelInfo(name="nv_argmax_final", opts_to_apply=()))

def nv_argmax(x:Tensor, axis:int=-1, keepdim:bool=True) -> Tensor:
  """Fused two-stage warp argmax with exact bitwise tie-breaking (lowest index wins)."""
  N = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or axis not in (-1, x.ndim - 1) or isinstance(N, UOp) or N % 32 != 0:
    return x.argmax(axis=axis, keepdim=keepdim)

  tile = 8 if N % 256 == 0 else 1
  elems = tile * 32
  num_blocks = N // elems
  orig_shape = x.shape
  x_flat = x.reshape(-1, N)
  tokens = x_flat.shape[0]

  pout = Tensor.empty(tokens, num_blocks, 32, dtype=dtypes.uint64, device=x.device)
  pout = Tensor.custom_kernel(pout, x_flat.contiguous(),
                              fxn=functools.partial(_argmax_partial_kernel, tokens=tokens, num_blocks=num_blocks, tile=tile))[0]

  out32 = Tensor.empty(tokens, 32, dtype=dtypes.int32, device=x.device)
  out32 = Tensor.custom_kernel(out32, pout,
                               fxn=functools.partial(_argmax_final_kernel, tokens=tokens, num_blocks=num_blocks))[0]
  res = out32[:, :1].contiguous()
  if not keepdim: res = res.squeeze(-1)
  return res.reshape(*orig_shape[:-1], *((1,) if keepdim else ()))
