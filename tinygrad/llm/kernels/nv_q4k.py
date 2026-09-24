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
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _load_lanes(xq[token, group, 0], 8)
    base = (output * in_features // GGML_BLOCK_SIZE + block) * Q4_WORDS
    qs_base = base + 4 + (subgroup//2)*8
    dot, qsum = UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
    for word_idx in range(8):
      word = (raw[qs_base+word_idx] >> ((subgroup & 1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
      dot = _nv_dp4a(word, xwords[word_idx], dot)
      qsum = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum)
    scale = (subgroup < 4).where(_load_byte(raw, base, 4 + subgroup) & 63,
      (_load_byte(raw, base, 8 + subgroup) & 15) | ((_load_byte(raw, base, subgroup) >> 6) << 4))
    minimum = (subgroup < 4).where(_load_byte(raw, base, 8 + subgroup) & 63,
      (_load_byte(raw, base, 8 + subgroup) >> 4) | ((_load_byte(raw, base, 4 + subgroup) >> 6) << 4))
    d = _half(raw[base] & 0xffff)
    dmin = _half((raw[base] >> 16).cast(dtypes.uint32) & 0xffff)
    return (dot.float()*d*scale.float() - qsum.float()*dmin*minimum.float()) * xd[token, group, 0]
  from tinygrad.llm.kernels.nv import _decode_linear
  return _decode_linear(out, out_features, group_count, group_dot, name="nv_linear_q4_k")


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