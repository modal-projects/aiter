#!/usr/bin/env python3
"""Check the pinned-register GEMM kernels: no compiler-emitted instruction may touch v128-v255 or an AGPR.

amdgpu_num_vgpr(128) is not honored once all 256 AGPRs are in use, so the compiler is free to put temporaries or its
SGPR-spill VGPR into v128+ and silently corrupt the pinned operand tiles. Inline-asm blocks (the pinned code
itself) are skipped.
  python3 check_pinned.py libhkmoe.s [kernel-name-regex]
"""
import re
import sys

src = open(sys.argv[1]).read()
pat = sys.argv[2] if len(sys.argv) > 2 else r"moe_gemm_kernel"
bad_total, checked = 0, 0
for m in re.finditer(r"^(_Z\w+):", src, re.M):
    fn = m.group(1)
    if not re.search(pat, fn):
        continue
    checked += 1
    body = src[m.start():]
    body = body[: body.index(".Lfunc_end")]
    inasm, bad = False, []
    for line in body.split("\n"):
        if ";;#ASMSTART" in line:
            inasm = True
            continue
        if ";;#ASMEND" in line:
            inasm = False
            continue
        t = line.split(";")[0].strip()
        if inasm or not t or t.startswith(".") or t.endswith(":"):
            continue
        regs = [int(x) for x in re.findall(r"\bv(\d+)\b", t)]
        for lo, hi in re.findall(r"\bv\[(\d+):(\d+)\]", t):
            regs += [int(lo), int(hi)]
        if any(r >= 128 for r in regs) or re.search(r"\ba\d+\b|\ba\[\d+:\d+\]|accvgpr", t):
            bad.append(t)
    bad_total += len(bad)
    print(f"{'OK ' if not bad else 'BAD'} {fn[:90]}: {len(bad)} compiler instrs touch pinned regs" + (f" e.g. {bad[:3]}" if bad else ""))
if not checked:
    sys.exit(f"no kernel matches {pat!r}")
sys.exit(1 if bad_total else 0)
