"""NVIDIA IQ4_XS decode GEMV.

IQ4_XS block: 256 weights = 136 bytes: d(2), scales_h(2), scales_l(4), qs(128).
The packed buffer is viewed as 34 uint32 words. Nonlinear IQ4 values are decoded with
CUDA byte permutation before signed dp4a accumulation.
"""
from __future__ import annotations
import functools
from typing import Any
from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.kernels.nv import Q8_GROUP_SIZE, _nv_dp4a, _q8_quantize, _decode_linear, _load_lanes, _half

IQ4_XS, GGML_BLOCK_SIZE, IQ4_WORDS = 23, 256, 34


IQ4_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


@functools.cache
def _iq4_lut(device:str) -> Tensor:
  return Tensor([x & 255 for x in IQ4_VALUES], dtype=dtypes.uint8, device=device).contiguous()


def _iq4_word(packed:UOp, shift:int, lut:UOp) -> UOp:
  word = UOp.const(0, dtypes.uint32)
  for i in range(4):
    idx = (packed >> (shift + 8*i)) & 15
    word |= lut[idx].cast(dtypes.uint32) << (8*i)
  return word

def _load_byte(raw:UOp, base:UOp, offset:UOp) -> UOp:
  return (raw[base + offset//4] >> ((offset&3)*8).cast(dtypes.uint32)) & 255


def _scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp]:
  low = _load_byte(raw, base, 4 + subgroup//2)
  scale = ((low >> (4*(subgroup%2)).cast(dtypes.uint32)) & 15) | ((((raw[base] >> 16) >> (2*subgroup).cast(dtypes.uint32)) & 3) << 4)
  return _half(raw[base] & 0xffff), (scale.cast(dtypes.uint8).bitcast(dtypes.int8)-32).float()


@functools.cache
def _iq4_xs_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, lut:UOp, out_features:int, in_features:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _load_lanes(xq[token, group, 0], 8)
    base = (output * in_features // GGML_BLOCK_SIZE + block) * IQ4_WORDS
    dot = UOp.const(0, dtypes.int32)
    for word_idx in range(8):
      packed = raw[base + 2 + subgroup*4 + word_idx%4]
      dot = _nv_dp4a(_iq4_word(packed, 4*(word_idx//4), lut), xwords[word_idx], dot)
    d, scale = _scales(raw, base, subgroup)
    return dot.float() * xd[token, group, 0] * d * scale
  return _decode_linear(out, out_features, group_count, group_dot, name="nv_linear_iq4_xs")


def iq4_xs_linear(layer:Any, x:Tensor) -> Tensor:
  assert layer.ggml_type == IQ4_XS and layer.in_features % Q8_GROUP_SIZE == 0
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  xq, xd = _q8_quantize(x, tokens, in_features)
  chunks = (in_features // Q8_GROUP_SIZE + 31) // 32
  out = Tensor.empty(tokens, out_features, chunks, 32, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xq.uop, xd.uop, _iq4_lut(x.device).uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i, src in enumerate(all_srcs))
  kernel = _iq4_xs_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))[..., 0]
  if chunks > 1: result = result.sum(-1)
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias
