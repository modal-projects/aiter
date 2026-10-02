# Block-FP8 grouped MoE (MI355X / gfx950)

`aiter.hk_fp8_moe` (JIT module `module_hk_fp8_moe`): a complete MoE layer forward (routing, both expert GEMMs,
SwiGLU, combine) for block-FP8 weights, built on a persistent 4-wave HipKittens GEMM that keeps its operand tiles and
accumulators in pinned registers and applies every FP8 scale with `V_MFMA_SCALE`. C API in `hk_moe.h`, called through
the ctypes entry points in `hk_fp8_moe_entry.cu`. The same sources live in HipKittens
(`kernels/cdna4/moe/fp8_grouped`), where they build standalone.

## Contract

| | |
|---|---|
| shapes | hidden 4096, intermediate 1024 (GLM-5.3-Flash expert shard at TP2), top-k 8 or 9, up to 1024 experts; anything else returns an error code |
| weights | fp8 e4m3, `w13` [E][2*1024][4096] (gate rows then up rows), `w2` [E][4096][1024], in aiter's `shuffle_weight(w, (16, 16))` layout |
| weight scales | fp32 128x128 block scales, **powers of two** (UE8M0; only the exponent is read). Checkpoints with arbitrary block scales must be requantized at load. |
| activations | bf16 in / out; quantized inside to e4m3 with a power-of-two scale per (token, 128 columns) |
| expert | `silu(min(gate, L)) * clamp(up, -L, L)` for `swiglu_limit` L > 0, else `silu(gate) * up` |
| workspace | `hk_moe_workspace_bytes(T, topk, E)` bytes; everything is enqueued on the given stream (graph-capturable) |

This is the numerics recipe SGLang uses for DeepGEMM MoE on B300 (UE8M0 weights, pow2 activation scales).

## Pipeline

Seven launches (plus a memset of the expert counts):

1. `count_and_quant`: per-expert selection counts, and in the same launch the token-major FP8 quantization of x
   (pow2 scale per 128 columns).
2. `moe_scan` / `moe_scatter` sort the token selections by expert into 256-row tiles (`moe_scan` also marks the
   padding rows).
3. `build_gather_tables`: per-thread gather offsets, so stage 1 reads its expert-sorted A rows straight from the
   token-major buffer (no sorted copy), and stage 1's packed MFMA A-scale words.
4. `moe_gemm_kernel<2048, 4096, true>`: gate/up GEMM; the epilogue applies SwiGLU, requantizes the activations to FP8
   and writes their scales already packed for stage 2.
5. `moe_gemm_kernel<4096, 1024, false>`: down GEMM, unweighted bf16 output per sorted row. Rows that only pad an
   expert to whole tiles are not written (their stores go to an out-of-range buffer offset, which keeps the kernel's
   vmcnt accounting).
6. `moe_reduce_w`: per-token weighted sum of its top-k rows.

The GEMM is persistent: each workgroup strides over 256x256 output tiles and its K-step pipeline (two steps in flight)
runs across tile boundaries, so a tile's last steps already load the next tile's first ones. The down GEMM stores each
finished accumulator inside the next tile's first K-step, so the stores overlap MFMAs. When experts average at most 144
rows (below ~5k tokens), both GEMMs use a build that skips the MFMAs of tile halves holding only padding rows.

## Files

| file | |
|---|---|
| `hk_moe.h` | C API |
| `hk_moe.cu` | the whole kernel, in sections: tile-load helpers, pinned-register tiles (operand / accumulator art tiles, accumulator access, bf16 tile store), global -> LDS tile loads and the K-step MFMA/load cluster, the grouped GEMM and its stage-1 epilogue, routing and the weighted combine, activation quantization and scale tables, and the host side (workspace layout, launch sequence) |
| `hk_fp8_moe_entry.cu` | ctypes entry points (forward, workspace size) used by `aiter/ops/hk_fp8_moe.py` |
| `check_pinned.py` | verifies no compiler-generated instruction touches the pinned registers |

## Build and test

aiter JIT-builds the module on first use, with HipKittens headers from `HIP_KITTENS_DIR` (fetched at the pinned
commit by default). `op_tests/test_hk_fp8_moe.py` checks against an fp32 reference (top-8 and top-9 with the shared
expert fused, with and without the SwiGLU clamp, 1 to 16384 tokens) and prints timings.

Verified with ROCm 7.0 (clang 20) and ROCm 7.2.4 (clang 22): bitwise-identical outputs, same speed, no spills.
`amdgpu_num_vgpr(128)` is not honored once all 256 AGPRs are in use, so a code or compiler change that raises register
pressure can silently corrupt the operand tiles. After every kernel or compiler change, check the device assembly for
VGPR spills and run `check_pinned.py` on it:

```bash
hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -I$HIP_KITTENS_DIR/include --cuda-device-only -S \
    hk_moe.cu -o hk_moe.s
grep -E 'vgpr_spill_count: [1-9]' hk_moe.s; python3 check_pinned.py hk_moe.s
```

## Performance

MI355X, one GPU, 288 routed experts, median of 20 with HipKittens' `test_hk_moe.py` (random weights):

| tokens | top-k 8 | top-k 9 (shared expert fused) |
|---:|---:|---:|
| 4096 | 1.02 ms | 1.09 ms |
| 8192 | 1.26 ms | 1.36 ms |
| 16384 | 2.20 ms | 2.35 ms |

For reference, aiter's block-FP8 fused MoE takes 4.0 ms and its MXFP8 path 2.8 ms at 16384 tokens on the same weights.
Below ~2k tokens the layer is weight-bandwidth bound and fused_moe with tuned tables is faster (end to end in SGLang
the crossover was ~1.5k tokens). Relative error vs the fp32 reference (same quantized weights) is ~4.7%, the same as
aiter's MXFP8 path.
