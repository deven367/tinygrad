"""NVIDIA Q4_K decode kernel — mirrors amd.py::_quant_decode_kernel with NV intrinsics.

Q4_K block: 256 weights = 144 bytes: d(2) dmin(2) scales(12) qs(128). 4-bit nibbles.
144 = 36 uint32 -> 4-byte aligned, safe for plain uint32 loads (no misaligned access).
Activation quantization is the SAME q8_quantize as the Q8_0 path (xq/xd 32-element groups).
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


@functools.cache
def _q4_k_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
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

    sc_even = (subgroup_even < 4).where(_load_byte(raw, base, 4 + subgroup_even) & 63,
      (_load_byte(raw, base, 8 + subgroup_even) & 15) | ((_load_byte(raw, base, subgroup_even) >> 6) << 4))
    m_even = (subgroup_even < 4).where(_load_byte(raw, base, 8 + subgroup_even) & 63,
      (_load_byte(raw, base, 8 + subgroup_even) >> 4) | ((_load_byte(raw, base, 4 + subgroup_even) >> 6) << 4))

    sc_odd = (subgroup_odd < 4).where(_load_byte(raw, base, 4 + subgroup_odd) & 63,
      (_load_byte(raw, base, 8 + subgroup_odd) & 15) | ((_load_byte(raw, base, subgroup_odd) >> 6) << 4))
    m_odd = (subgroup_odd < 4).where(_load_byte(raw, base, 8 + subgroup_odd) & 63,
      (_load_byte(raw, base, 8 + subgroup_odd) >> 4) | ((_load_byte(raw, base, 4 + subgroup_odd) >> 6) << 4))

    d = _half(raw[base] & 0xffff)
    dmin = _half((raw[base] >> 16).cast(dtypes.uint32) & 0xffff)

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
  kernel = _q4_k_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))[..., 0]
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias