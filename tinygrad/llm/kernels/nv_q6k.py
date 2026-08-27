"""NVIDIA Q6_K decode kernel — mirrors amd.py::_quant_decode_kernel with NV intrinsics.

Q6_K block: 256 weights = 210 bytes: ql(128) qh(64) scales(16 int8) d(2, f16 at byte 208).
6 bits/weight = low 4 bits (ql, 2 weights/byte) | high 2 bits (qh, 4 weights/byte, 2-bit slots).
16 scales per block: weight w uses scales[w//16]. q = (low | (high << 4)) - 32; val = q*s*d.

210 bytes = 105 uint16 words. Block base is 210*k: always 2-aligned, never guaranteed
4-aligned -> the packed buffer is viewed as uint16 and every weight access is a 16-bit load.

Activation quantization is the SAME _q8_quantize as the Q8_0/Q4_K paths (32-elem groups).

Bit layout verified three ways: llama.cpp ggml-quants.c dequantize_row_q6_K (reference),
tinygrad gguf.py loader bit positions, and amd.py::_quant_decode_kernel Q6_K branch.
"""
from __future__ import annotations
import functools
from typing import Any
from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.kernels.nv import (Q8_GROUP_SIZE, _nv_dp4a, _nv_ldcs16, _q8_quantize,
                                     _decode_linear, _load_lanes, _half)

Q6_K, GGML_BLOCK_SIZE, Q6_WORDS = 14, 256, 105   # 210 bytes = 105 u16 words per block


@functools.cache
def _q6_k_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _load_lanes(xq[token, group, 0], 8)
    base = (output * in_features // GGML_BLOCK_SIZE + block) * Q6_WORDS
    # 4 weights per word_idx, 8 word_idx per 32-weight subgroup:
    #   ql bytes: h*64 + (m%64) .. +3  (h=w//128, m=w%128)
    #   qh bytes: 128 + h*32 + (m%32) .. +3, 2-bit slot (m//32)
    ql_base = 16 * (subgroup % 2) + 32 * (subgroup // 4)
    qh_base = 64 + 16 * (subgroup // 4)
    nib = ((subgroup & 2) // 2) * 4      # ql nibble shift: 0 for s in {0,1,4,5}, 4 for {2,3,6,7}
    qsh = 2 * (subgroup % 4)             # qh 2-bit slot: 0/2/4/6
    dots = [UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)]
    qsums = [UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)]
    for word_idx in range(8):
      ql = _nv_ldcs16(raw[base + ql_base + 2*word_idx]) | \
        (_nv_ldcs16(raw[base + ql_base + 2*word_idx + 1]).cast(dtypes.uint32) << 16)
      qh = _nv_ldcs16(raw[base + qh_base + 2*word_idx]) | \
        (_nv_ldcs16(raw[base + qh_base + 2*word_idx + 1]).cast(dtypes.uint32) << 16)
      qword = ((ql >> nib.cast(dtypes.uint32)) & 0x0f0f0f0f) | (((qh >> qsh.cast(dtypes.uint32)) & 0x03030303) << 4)
      c = word_idx // 4                  # weights 0-15 -> scales[2s], 16-31 -> scales[2s+1]
      dots[c] = _nv_dp4a(qword, xwords[word_idx], dots[c])
      qsums[c] = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsums[c])
    sw = _nv_ldcs16(raw[base + 96 + subgroup])  # scales[2s] | scales[2s+1] << 8
    s0 = (sw & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float()
    s1 = (sw >> 8).cast(dtypes.uint8).bitcast(dtypes.int8).float()
    d = _half(_nv_ldcs16(raw[base + 104]))
    return ((dots[0].float() - 32.0 * qsums[0].float()) * s0 + (dots[1].float() - 32.0 * qsums[1].float()) * s1) \
      * xd[token, group, 0] * d
  return _decode_linear(out, out_features, group_count, group_dot, name="nv_linear_q6_k")


def q6_k_linear(layer:Any, x:Tensor) -> Tensor:
  assert layer.ggml_type == Q6_K and layer.in_features % Q8_GROUP_SIZE == 0
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  xq, xd = _q8_quantize(x, tokens, in_features)
  chunks = (in_features // Q8_GROUP_SIZE + 31) // 32
  out = Tensor.empty(tokens, out_features, chunks, 32, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xq.uop, xd.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i, src in enumerate(all_srcs))
  kernel = _q6_k_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))[..., 0]
  if chunks > 1: result = result.sum(-1)
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias
