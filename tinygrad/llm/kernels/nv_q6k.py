"""NVIDIA Q6_K decode kernel — cooperative warp GEMV with NV intrinsics.

Q6_K block: 256 weights = 210 bytes: ql(128) qh(64) scales(16 int8) d(2, f16 at byte 208).
6 bits/weight = low 4 bits (ql, 2 weights/byte) | high 2 bits (qh, 4 weights/byte, 2-bit slots).
16 scales per block: weight w uses scales[w//16]. q = (low | (high << 4)) - 32; val = q*s*d.

210 bytes = 105 uint16 words. Block base is 210*k: always 2-aligned, never guaranteed
4-aligned -> the packed buffer is viewed as uint16 and every weight access is a 16-bit load.
To avoid misaligned 32-bit loads, 32-bit words are formed from two uint16 loads.

Each warp (32 threads) processes 1 output feature across all input blocks.
The 32 threads cooperatively load the 128-byte ql and 64-byte qh arrays.
"""
from __future__ import annotations
import functools
from typing import Any
from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.nv import (Q8_GROUP_SIZE, _nv_dp4a, _nv_shuffle_xor, _nv_fmax,
                                     _q8_quantize, _half)

Q6_K, GGML_BLOCK_SIZE, Q6_WORDS = 14, 256, 105   # 210 bytes = 105 u16 words per block


def _warp_reduce(value:UOp, maximum:bool=False) -> UOp:
  for offset in (16, 8, 4, 2, 1):
    other = _nv_shuffle_xor(value, offset)
    value = _nv_fmax(value, other) if maximum else value + other
  return value


def _u16_word(raw:UOp, idx:UOp) -> UOp:
  lo = raw[idx].cast(dtypes.uint32)
  hi = raw[idx + 1].cast(dtypes.uint32)
  return lo | (hi << 16)


@functools.cache
def _q6_k_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  num_blocks = in_features // GGML_BLOCK_SIZE
  token_output = UOp.range(out.shape[0]*out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token = token_output // out_features
  output = token_output % out_features

  half = lane >> 4          # 0 for lanes 0..15, 1 for lanes 16..31
  quarter = (lane >> 3) & 1 # 0 for lanes 0..7, 16..23; 1 for lanes 8..15, 24..31
  w = lane & 7              # 0..7

  s0 = (half << 2) + quarter      # subgroups 0, 1, 4, 5
  s1 = s0 + 2                     # subgroups 2, 3, 6, 7

  acc = UOp.const(0, dtypes.float32)

  for b in range(num_blocks):
    base = (output * num_blocks + b) * Q6_WORDS

    # ql: 128 bytes = 64 u16 words. Thread lane in 0..31 loads word 2*lane
    w_ql = _u16_word(raw, base + (lane << 1))

    # qh: 64 bytes = 32 u16 words, starts at offset 64 in u16.
    # Half 0 uses u16 offset 64 + 2*w, half 1 uses u16 offset 64 + 16 + 2*w
    w_qh = _u16_word(raw, base + 64 + (half << 4) + (w << 1))

    # Extract 2-bit high bits
    sh0 = quarter << 1      # 0 or 2
    sh1 = sh0 + 4           # 4 or 6
    qh0 = (w_qh >> sh0.cast(dtypes.uint32)) & 0x03030303
    qh1 = (w_qh >> sh1.cast(dtypes.uint32)) & 0x03030303

    ql0 = w_ql & 0x0f0f0f0f
    ql1 = (w_ql >> 4) & 0x0f0f0f0f

    qword0 = ql0 | (qh0 << 4)
    qword1 = ql1 | (qh1 << 4)

    # Activations
    grp0 = b * 8 + s0
    grp1 = b * 8 + s1

    x0 = xq[token, grp0, w].load()
    x1 = xq[token, grp1, w].load()

    dot0 = _nv_dp4a(qword0, x0, UOp.const(0, dtypes.int32))
    qsum0 = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x0, UOp.const(0, dtypes.int32))

    dot1 = _nv_dp4a(qword1, x1, UOp.const(0, dtypes.int32))
    qsum1 = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x1, UOp.const(0, dtypes.int32))

    # Scales: 16 int8 scales at u16 offset 96 (8 u16 words).
    # Each u16 holds 2 int8 scales: scales[2*s] | (scales[2*s+1] << 8).
    sw0 = raw[base + 96 + s0]
    sw1 = raw[base + 96 + s1]

    # w < 4 uses lower 8 bits (scales[2*s]), w >= 4 uses upper 8 bits (scales[2*s+1])
    sc0 = (w < 4).where(sw0 & 255, sw0 >> 8).cast(dtypes.uint8).bitcast(dtypes.int8).float()
    sc1 = (w < 4).where(sw1 & 255, sw1 >> 8).cast(dtypes.uint8).bitcast(dtypes.int8).float()

    xd0 = xd[token, grp0, 0]
    xd1 = xd[token, grp1, 0]

    d = _half(raw[base + 104])

    val0 = (dot0.float() - 32.0 * qsum0.float()) * sc0 * xd0
    val1 = (dot1.float() - 32.0 * qsum1.float()) * sc1 * xd1

    acc = acc + (val0 + val1) * d

  total = _warp_reduce(acc)
  return out[token, output, lane].store(total.cast(out.dtype)).end(token_output, lane).sink(
    arg=KernelInfo(name="nv_linear_q6_k", opts_to_apply=()))


def q6_k_linear(layer:Any, x:Tensor) -> Tensor:
  assert layer.ggml_type == Q6_K and layer.in_features % GGML_BLOCK_SIZE == 0
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  xq, xd = _q8_quantize(x, tokens, in_features)
  out = Tensor.empty(tokens, out_features, 32, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xq.uop, xd.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i, src in enumerate(all_srcs))
  kernel = _q6_k_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))[..., 0]
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias
