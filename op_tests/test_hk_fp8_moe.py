# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the HipKittens block-FP8 grouped MoE against an fp32 torch reference.

Random experts with power-of-two 128x128 block scales (the kernel's contract), top-8 routed or top-9 with the shared
expert fused as expert 288, with and without the SwiGLU clamp. Token counts cover the small-batch variant that skips
all-padding tile halves (< ~5k tokens) and the large-batch one.
"""

import argparse
import sys

import torch

import aiter
from aiter.ops.hk_fp8_moe import HIDDEN, INTER, hk_fp8_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import perftest

FP8 = torch.float8_e4m3fn
# power-of-two activation scales on top of the shared fp8 weights
TOL = 0.06


def requant_pow2(w):
    """[E, n, k] float -> fp8 with power-of-two 128x128 block scales [E, n/128, k/128]."""
    E, n, k = w.shape
    b = w.float().view(E, n // 128, 128, k // 128, 128)
    s = torch.exp2(
        torch.ceil(
            torch.log2(b.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / 448.0)
        )
    )
    return (b / s).to(FP8).view(E, n, k), s.view(E, n // 128, k // 128).contiguous()


def dequant(q, s):
    E, n, k = q.shape
    b = q.float().view(E, n // 128, 128, k // 128, 128)
    return (b * s.view(E, n // 128, 1, k // 128, 1)).view(E, n, k)


def reference(x, W13, W2, tw, ids, limit):
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for e in torch.unique(ids).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        g, u = (x[tok].float() @ W13[e].t()).chunk(2, dim=-1)
        if limit > 0:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        act = torch.nn.functional.silu(g) * u
        out.index_add_(0, tok, act @ W2[e].t() * tw[tok, slot, None])
    return out


def make_weights(E):
    gen = torch.Generator(device="cuda").manual_seed(0)
    w13, s13 = requant_pow2(
        torch.randn(E, 2 * INTER, HIDDEN, device="cuda", generator=gen) * 0.02
    )
    w2, s2 = requant_pow2(
        torch.randn(E, HIDDEN, INTER, device="cuda", generator=gen) * 0.02
    )
    return w13, s13, w2, s2


def make_routing(T, E):
    gen = torch.Generator(device="cuda").manual_seed(T)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
    tw, ids = torch.topk(
        torch.randn(T, 288, device="cuda", generator=gen).softmax(-1), 8, dim=-1
    )
    if E == 289:  # the fused shared expert, selected by every token
        ids = torch.cat([ids, torch.full_like(ids[:, :1], 288)], dim=1)
        tw = torch.cat([tw, torch.ones_like(tw[:, :1])], dim=1)
    return x, ids.to(torch.int32), tw


@perftest()
def run(x, ids, tw, w13, s13, w2, s2, limit):
    return hk_fp8_moe(x, ids, tw, w13, s13, w2, s2, swiglu_limit=limit)


def test_case(T, E, limit, weights):
    w13, s13, w2, s2 = weights
    x, ids, tw = make_routing(T, E)
    out, us = run(
        x,
        ids,
        tw,
        shuffle_weight(w13, (16, 16)),
        s13,
        shuffle_weight(w2, (16, 16)),
        s2,
        limit,
    )
    rows = torch.arange(0, T, max(1, T // 256), device="cuda")
    ref = reference(
        x[rows], dequant(w13, s13), dequant(w2, s2), tw[rows], ids[rows], limit
    )
    err = ((out[rows].float() - ref).norm() / ref.norm()).item()
    ok = err < TOL and not torch.isnan(out).any()
    aiter.logger.info(
        f"hk_fp8_moe T={T:6d} experts={E} topk={ids.shape[1]} limit={limit:4.1f}: rel err {err:.4f} "
        f"{us:8.1f} us  {'ok' if ok else 'FAIL'}"
    )
    return ok


def test_rejects(weights):
    w13, s13, w2, s2 = weights
    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(64, 4, dtype=torch.int32, device="cuda")
    ok = True
    for bad in (
        lambda: hk_fp8_moe(x, ids, ids.float(), w13, s13, w2, s2),  # top-k 4
        lambda: hk_fp8_moe(
            x[:, :2048], ids.repeat(1, 2), ids.float().repeat(1, 2), w13, s13, w2, s2
        ),
    ):
        try:
            bad()
            ok = False
        except ValueError:
            pass
    aiter.logger.info(
        f"hk_fp8_moe rejects unsupported shapes: {'ok' if ok else 'FAIL'}"
    )
    return ok


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="HipKittens block-FP8 grouped MoE: correctness vs fp32 and timing",
)
parser.add_argument(
    "-t", "--tokens", type=int, nargs="*", default=None, help="Token counts"
)
args = parser.parse_args()

if aiter.get_gfx() != "gfx950":
    print(f"Skipping test_hk_fp8_moe.py: requires gfx950, got {aiter.get_gfx()}")
    sys.exit(0)

tokens = args.tokens or [1, 1024, 4096, 6144, 16384]
results = []
for E in (288, 289):
    weights = make_weights(E)
    for T in tokens:
        for limit in (0.0, 10.0):
            results.append(test_case(T, E, limit, weights))
    results.append(test_rejects(weights))
    del weights
sys.exit(0 if all(results) else 1)
