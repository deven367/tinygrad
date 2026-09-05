"""NVIDIA Q5_K decode GEMV.

Q5_K block: 256 weights = 176 bytes: d(2), dmin(2), scales(12), qh(32), qs(128).
The packed buffer is viewed as 44 uint32 words. Activations use the shared Q8 quantizer.
"""
from __future__ import annotations
import functools
from typing import Any
from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.kernels.nv import Q8_GROUP_SIZE, _nv_dp4a, _q8_quantize, _decode_linear, _load_lanes, _half

Q5_K, GGML_BLOCK_SIZE, Q5_WORDS = 13, 256, 44


def _load_byte(raw:UOp, base:UOp, offset:UOp) -> UOp:
  return (raw[base + offset//4] >> ((offset&3)*8).cast(dtypes.uint32)) & 255


def _scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  scale = (subgroup < 4).where(_load_byte(raw, base, 4 + subgroup) & 63,
    (_load_byte(raw, base, 8 + subgroup) & 15) | ((_load_byte(raw, base, subgroup) >> 6) << 4))
  minimum = (subgroup < 4).where(_load_byte(raw, base, 8 + subgroup) & 63,
    (_load_byte(raw, base, 8 + subgroup) >> 4) | ((_load_byte(raw, base, 4 + subgroup) >> 6) << 4))
  return _half(raw[base] & 0xffff), _half((raw[base] >> 16) & 0xffff), scale.float(), minimum.float()


@functools.cache
def _q5_k_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _load_lanes(xq[token, group, 0], 8)
    base = (output * in_features // GGML_BLOCK_SIZE + block) * Q5_WORDS
    qs_base = base + 12 + (subgroup//2)*8
    dot, qsum = UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
    for word_idx in range(8):
      word = (raw[qs_base+word_idx] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
      word |= (((raw[base+4+word_idx] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4)
      dot = _nv_dp4a(word, xwords[word_idx], dot)
      qsum = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum)
    d, dmin, scale, minimum = _scales(raw, base, subgroup)
    return (dot.float()*d*scale - qsum.float()*dmin*minimum) * xd[token, group, 0]
  return _decode_linear(out, out_features, group_count, group_dot, name="nv_linear_q5_k")


def q5_k_linear(layer:Any, x:Tensor) -> Tensor:
  assert layer.ggml_type == Q5_K and layer.in_features % Q8_GROUP_SIZE == 0
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  xq, xd = _q8_quantize(x, tokens, in_features)
  chunks = (in_features // Q8_GROUP_SIZE + 31) // 32
  out = Tensor.empty(tokens, out_features, chunks, 32, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xq.uop, xd.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i, src in enumerate(all_srcs))
  kernel = _q5_k_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))[..., 0]
  if chunks > 1: result = result.sum(-1)
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias
