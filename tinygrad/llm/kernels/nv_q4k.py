"""NVIDIA Q4_K decode kernel — mirrors amd.py::_quant_decode_kernel with NV intrinsics.

Q4_K block: 256 weights = 144 bytes: d(2) dmin(2) scales(12) qs(128). 4-bit nibbles.
144 = 36 uint32 -> 4-byte aligned, safe for plain uint32 loads (no misaligned access).
Activation quantization is the SAME q8_quantize as the Q8_0 path (xq/xd 32-element groups).

Optimizations:
- 128-bit vectorized memory loads (uint4 / v4.u32) for num_blocks % 4 == 0.
- 64-bit vectorized memory loads (uint2 / v2.u32) for num_blocks % 2 == 0.
- Fallback cooperative warp GEMV for odd num_blocks.
"""
from __future__ import annotations
import functools
from typing import Any
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
import math
from tinygrad.llm.kernels.nv import (Q8_GROUP_SIZE, _nv_dp4a, _nv_shuffle_xor, _nv_fmax,
                                     nv_custom_kernels_supported, _q8_quantize)

Q4_K, GGML_BLOCK_SIZE, Q4_WORDS = 12, 256, 36


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


def _load_byte(raw:UOp, base:UOp, offset:int) -> UOp:
  return (raw[base + offset//4] >> ((offset & 3)*8).cast(dtypes.uint32)) & 255


def _half(value:UOp) -> UOp: return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()


def _q4k_scales(hdr0:UOp, hdr1:UOp, hdr2:UOp, hdr3:UOp, pair:UOp) -> tuple[UOp, UOp, UOp, UOp, UOp, UOp]:
  shift_even = ((pair & 1) * 16).cast(dtypes.uint32)
  shift_odd  = shift_even + 8
  sc_even = (pair < 2).where((hdr1 >> shift_even) & 63, ((hdr3 >> shift_even) & 15) | ((((hdr1 >> shift_even) & 255) >> 6) << 4))
  m_even  = (pair < 2).where((hdr2 >> shift_even) & 63, (((hdr3 >> shift_even) & 255) >> 4) | ((((hdr2 >> shift_even) & 255) >> 6) << 4))
  sc_odd  = (pair < 2).where((hdr1 >> shift_odd) & 63, ((hdr3 >> shift_odd) & 15) | ((((hdr1 >> shift_odd) & 255) >> 6) << 4))
  m_odd   = (pair < 2).where((hdr2 >> shift_odd) & 63, (((hdr3 >> shift_odd) & 255) >> 4) | ((((hdr2 >> shift_odd) & 255) >> 6) << 4))
  d = _half(hdr0 & 0xffff)
  dmin = _half((hdr0 >> 16).cast(dtypes.uint32) & 0xffff)
  return d, dmin, sc_even, m_even, sc_odd, m_odd


@functools.cache
def _q4_k_v4_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  """128-bit vectorized cooperative warp kernel: 8 threads/block, 4 words/thread (uint4)."""
  num_blocks = in_features // GGML_BLOCK_SIZE
  assert num_blocks % 4 == 0, f"num_blocks must be divisible by 4, got {num_blocks}"
  token_output = UOp.range(out.shape[0]*out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token = token_output // out_features
  output = token_output % out_features

  lane8 = lane & 7
  block_in_quad = lane >> 3
  pair = lane8 >> 1
  k = lane8 & 1
  subgroup_even = pair * 2
  subgroup_odd = pair * 2 + 1

  w0_idx = k * 4
  w1_idx = k * 4 + 1
  w2_idx = k * 4 + 2
  w3_idx = k * 4 + 3

  acc = UOp.const(0, dtypes.float32)

  for b_quad in range(num_blocks // 4):
    b = b_quad * 4 + block_in_quad
    base = (output * num_blocks + b) * Q4_WORDS

    hdr0 = raw[base]
    hdr1 = raw[base + 1]
    hdr2 = raw[base + 2]
    hdr3 = raw[base + 3]

    raw_idx = base + 4 + pair * 8 + k * 4
    w0 = raw[raw_idx]
    w1 = raw[raw_idx + 1]
    w2 = raw[raw_idx + 2]
    w3 = raw[raw_idx + 3]

    grp_even = b * 8 + subgroup_even
    grp_odd = b * 8 + subgroup_odd

    x0_even = xq[token, grp_even, w0_idx].load()
    x1_even = xq[token, grp_even, w1_idx].load()
    x2_even = xq[token, grp_even, w2_idx].load()
    x3_even = xq[token, grp_even, w3_idx].load()

    x0_odd  = xq[token, grp_odd,  w0_idx].load()
    x1_odd  = xq[token, grp_odd,  w1_idx].load()
    x2_odd  = xq[token, grp_odd,  w2_idx].load()
    x3_odd  = xq[token, grp_odd,  w3_idx].load()

    w0_even = w0 & 0x0f0f0f0f
    w0_odd  = (w0 >> 4) & 0x0f0f0f0f
    w1_even = w1 & 0x0f0f0f0f
    w1_odd  = (w1 >> 4) & 0x0f0f0f0f
    w2_even = w2 & 0x0f0f0f0f
    w2_odd  = (w2 >> 4) & 0x0f0f0f0f
    w3_even = w3 & 0x0f0f0f0f
    w3_odd  = (w3 >> 4) & 0x0f0f0f0f

    dot_even = _nv_dp4a(w0_even, x0_even, UOp.const(0, dtypes.int32))
    dot_even = _nv_dp4a(w1_even, x1_even, dot_even)
    dot_even = _nv_dp4a(w2_even, x2_even, dot_even)
    dot_even = _nv_dp4a(w3_even, x3_even, dot_even)

    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x0_even, UOp.const(0, dtypes.int32))
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x1_even, qsum_even)
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x2_even, qsum_even)
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x3_even, qsum_even)

    dot_odd = _nv_dp4a(w0_odd, x0_odd, UOp.const(0, dtypes.int32))
    dot_odd = _nv_dp4a(w1_odd, x1_odd, dot_odd)
    dot_odd = _nv_dp4a(w2_odd, x2_odd, dot_odd)
    dot_odd = _nv_dp4a(w3_odd, x3_odd, dot_odd)

    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x0_odd, UOp.const(0, dtypes.int32))
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x1_odd, qsum_odd)
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x2_odd, qsum_odd)
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x3_odd, qsum_odd)

    d, dmin, sc_even, m_even, sc_odd, m_odd = _q4k_scales(hdr0, hdr1, hdr2, hdr3, pair)

    xd_even = xd[token, grp_even, 0]
    xd_odd  = xd[token, grp_odd,  0]

    val_even = (dot_even.float()*d*sc_even.float() - qsum_even.float()*dmin*m_even.float()) * xd_even
    val_odd  = (dot_odd.float() *d*sc_odd.float()  - qsum_odd.float() *dmin*m_odd.float() ) * xd_odd

    acc = acc + val_even + val_odd

  total = _warp_reduce(acc)
  return out[token, output, lane].store(total.cast(out.dtype)).end(token_output, lane).sink(
    arg=KernelInfo(name="nv_linear_q4_k_v4", opts_to_apply=()))


@functools.cache
def _q4_k_v2_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  """64-bit vectorized cooperative warp kernel: 16 threads/block, 2 words/thread (uint2)."""
  num_blocks = in_features // GGML_BLOCK_SIZE
  assert num_blocks % 2 == 0, f"num_blocks must be even, got {num_blocks}"
  token_output = UOp.range(out.shape[0]*out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token = token_output // out_features
  output = token_output % out_features

  lane16 = lane & 15
  block_in_pair = lane >> 4
  pair = lane16 >> 2
  k = lane16 & 3
  subgroup_even = pair * 2
  subgroup_odd = pair * 2 + 1

  w0_idx = k * 2
  w1_idx = k * 2 + 1

  acc = UOp.const(0, dtypes.float32)

  for b_pair in range(num_blocks // 2):
    b = b_pair * 2 + block_in_pair
    base = (output * num_blocks + b) * Q4_WORDS

    hdr0 = raw[base]
    hdr1 = raw[base + 1]
    hdr2 = raw[base + 2]
    hdr3 = raw[base + 3]

    raw_idx = base + 4 + pair * 8 + k * 2
    w0 = raw[raw_idx]
    w1 = raw[raw_idx + 1]

    grp_even = b * 8 + subgroup_even
    grp_odd = b * 8 + subgroup_odd

    x0_even = xq[token, grp_even, w0_idx].load()
    x1_even = xq[token, grp_even, w1_idx].load()
    x0_odd  = xq[token, grp_odd,  w0_idx].load()
    x1_odd  = xq[token, grp_odd,  w1_idx].load()

    w0_even = w0 & 0x0f0f0f0f
    w0_odd  = (w0 >> 4) & 0x0f0f0f0f
    w1_even = w1 & 0x0f0f0f0f
    w1_odd  = (w1 >> 4) & 0x0f0f0f0f

    dot_even = _nv_dp4a(w0_even, x0_even, UOp.const(0, dtypes.int32))
    dot_even = _nv_dp4a(w1_even, x1_even, dot_even)
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x0_even, UOp.const(0, dtypes.int32))
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x1_even, qsum_even)

    dot_odd = _nv_dp4a(w0_odd, x0_odd, UOp.const(0, dtypes.int32))
    dot_odd = _nv_dp4a(w1_odd, x1_odd, dot_odd)
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x0_odd, UOp.const(0, dtypes.int32))
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x1_odd, qsum_odd)

    d, dmin, sc_even, m_even, sc_odd, m_odd = _q4k_scales(hdr0, hdr1, hdr2, hdr3, pair)

    xd_even = xd[token, grp_even, 0]
    xd_odd  = xd[token, grp_odd,  0]

    val_even = (dot_even.float()*d*sc_even.float() - qsum_even.float()*dmin*m_even.float()) * xd_even
    val_odd  = (dot_odd.float() *d*sc_odd.float()  - qsum_odd.float() *dmin*m_odd.float() ) * xd_odd

    acc = acc + val_even + val_odd

  total = _warp_reduce(acc)
  return out[token, output, lane].store(total.cast(out.dtype)).end(token_output, lane).sink(
    arg=KernelInfo(name="nv_linear_q4_k_v2", opts_to_apply=()))


@functools.cache
def _q4_k_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  """Cooperative warp fallback kernel: 32 threads/block, 1 word/thread."""
  num_blocks = in_features // GGML_BLOCK_SIZE
  token_output = UOp.range(out.shape[0]*out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token = token_output // out_features
  output = token_output % out_features

  pair = lane >> 3
  w_idx = lane & 7
  subgroup_even = pair * 2
  subgroup_odd = pair * 2 + 1

  acc = UOp.const(0, dtypes.float32)

  for b in range(num_blocks):
    base = (output * num_blocks + b) * Q4_WORDS

    hdr0 = raw[base]
    hdr1 = raw[base + 1]
    hdr2 = raw[base + 2]
    hdr3 = raw[base + 3]

    w = raw[base + 4 + lane]

    grp_even = b * 8 + subgroup_even
    grp_odd = b * 8 + subgroup_odd

    x_even = xq[token, grp_even, w_idx].load()
    x_odd  = xq[token, grp_odd,  w_idx].load()

    w_even = w & 0x0f0f0f0f
    w_odd  = (w >> 4) & 0x0f0f0f0f

    dot_even = _nv_dp4a(w_even, x_even, UOp.const(0, dtypes.int32))
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x_even, UOp.const(0, dtypes.int32))

    dot_odd = _nv_dp4a(w_odd, x_odd, UOp.const(0, dtypes.int32))
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x_odd, UOp.const(0, dtypes.int32))

    d, dmin, sc_even, m_even, sc_odd, m_odd = _q4k_scales(hdr0, hdr1, hdr2, hdr3, pair)

    xd_even = xd[token, grp_even, 0]
    xd_odd  = xd[token, grp_odd,  0]

    val_even = (dot_even.float()*d*sc_even.float() - qsum_even.float()*dmin*m_even.float()) * xd_even
    val_odd  = (dot_odd.float() *d*sc_odd.float()  - qsum_odd.float() *dmin*m_odd.float() ) * xd_odd

    acc = acc + val_even + val_odd

  total = _warp_reduce(acc)
  return out[token, output, lane].store(total.cast(out.dtype)).end(token_output, lane).sink(
    arg=KernelInfo(name="nv_linear_q4_k", opts_to_apply=()))


def q4_k_linear(layer:Any, x:Tensor) -> Tensor:
  assert layer.ggml_type == Q4_K and layer.in_features % Q8_GROUP_SIZE == 0
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  xq, xd = _q8_quantize(x, tokens, in_features)
  out = Tensor.empty(tokens, out_features, 32, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xq.uop, xd.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
  num_blocks = in_features // GGML_BLOCK_SIZE
  if num_blocks % 4 == 0:
    kernel = _q4_k_v4_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  elif num_blocks % 2 == 0:
    kernel = _q4_k_v2_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  else:
    kernel = _q4_k_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))[..., 0]
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias