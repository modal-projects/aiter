// libhkmoe: block-FP8 MoE forward on MI355 (gfx950), HipKittens persistent 4-wave grouped GEMM.
//
//   out[t] = sum_j topk_w[t][j] * Expert_{topk_ids[t][j]}(x[t])
//   Expert_e(x) = W2_e (act(W13_e x)),  act = silu(min(gate, L)) * clamp(up, -L, L)  (L = swiglu_limit > 0)
//                                       act = silu(gate) * up                         (swiglu_limit <= 0)
//
// Shapes (the only ones compiled): hidden 4096, intermediate 1024 per GPU (GLM-5.3-Flash TP2 shard), top-k 8 or 9,
// up to 1024 experts. Built for large token counts (256-row tiles); small batches are faster on aiter's MoE.
//   x, out        bf16 [T][hidden]
//   topk_ids      int32 [T][topk]   (out-of-range ids contribute nothing)
//   topk_w        fp32  [T][topk]
//   w13           fp8 e4m3 [E][2*inter][hidden]  rows [0, inter) = gate, [inter, 2*inter) = up
//   w2            fp8 e4m3 [E][hidden][inter]
//   s13, s2       fp32 128x128 block scales [E][2*inter/128][hidden/128], [E][hidden/128][inter/128]
// Weights are in aiter's shuffle_weight(w, (16, 16)) layout. Every weight scale must be a power of two (UE8M0,
// e.g. requantized at load): only the exponent is used. Activations are quantized per (token, 128 columns) with
// power-of-two scales. The workspace must hold hk_moe_workspace_bytes(T, topk, E) bytes; all work is enqueued on
// `stream` (graph-capturable).
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum hk_moe_status {
    HK_MOE_OK = 0,
    HK_MOE_ERR_LAUNCH = -1,      // a kernel launch failed (hipGetLastError)
    HK_MOE_ERR_EXPERTS = -2,     // num_experts > 1024
    HK_MOE_ERR_WORKSPACE = -3,   // ws_bytes < hk_moe_workspace_bytes(T, topk, num_experts)
    HK_MOE_ERR_TOPK = -4,        // topk not 8 or 9
    HK_MOE_ERR_SHAPE = -5,       // hidden / inter not 4096 / 1024
};

size_t hk_moe_workspace_bytes(int T, int topk, int num_experts);

int hk_moe_forward(const void *x, void *out, int T, int hidden, int inter, const int *topk_ids, const float *topk_w,
                   int topk, const void *w13, const float *s13, const void *w2, const float *s2, int num_experts,
                   float swiglu_limit, void *workspace, size_t ws_bytes, hipStream_t stream);

#ifdef __cplusplus
}
#endif
