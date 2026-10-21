#!/usr/bin/env python3
"""Static schedule lint for the R4D prefill-attention key-tile loop (gfx1201).

Why this exists: the tile-16 double-buffered instance (O_DB) is a hand-scheduled pipeline, and a
compiler change -- not a source change -- once turned its per-tile prefetch prologue into two
separately-predicated regions with the address math duplicated and a full load drain between the two
global loads. No Python test can see that; it cost ~5.6 % of the kernel. This script reads the
emitted assembly of the paged attention translation unit and fails if the loop shape regresses, so a
bad schedule is caught at build time instead of at the end of a 47k-token run.

Input is the device assembly of `r4d_attn_paged_h256_gqa6.hip` (no objdump needed):

    hipcc -O3 -std=c++17 -fPIC --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off \\
          --offload-device-only -S r4d_attn_paged_h256_gqa6.hip -o attn.s
    python3 ci/attn_sched_lint.py attn.s            # exit 1 on any violation

Checks, with the thresholds of the measured reference loop
(research/paiton-vs-sly/v2/attn-db2/SPEC.md section 5.1):

    loop body, all paths              <= 372 instructions
    global loads                       2, in one block, <= 3 instructions apart
    a full drain between the two loads none (s_wait_loadcnt 0 between them)
    s_wait_loadcnt before the tail     <= 1 outstanding
    s_wait_kmcnt in the loop           <= 2
    s_mul_u64 in the loop              <= 1
    exec save/restore in the loop      <= 2 (one predicated prefetch region)
    VGPR / spill                       <= 240 / 0

The loop is the natural loop of the widest back edge that carries a barrier and the tile's WMMA
work, computed on the basic-block graph, so it does not care how the compiler lays the cold blocks
(mask skip, rescale, last-tile bodies, sunk predicated regions) out around the back edge.

Usage:  attn_sched_lint.py ASM [--tile N] [--mask 0xNN] [--json]
Exit:   0 = the instance passes, 1 = a violation, 2 = the instance is not in the input.
"""
import argparse
import json
import re
import sys

# _Z23r4d_attn_prefill_kernelILi24ELi16ELi256ELi6ELi16ELi0ELi8149611EEv7R4DArgs
KERNEL = re.compile(r"r4d_attn_prefill_kernelILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EE")
LABEL = re.compile(r"^([.A-Za-z_$][\w.$]*):")
INSTR = re.compile(r"^\t([a-z][a-z0-9_.]*)\b[ \t]*(.*)$")

O_DB = 4194304
O_QF32 = 16777216
LIMITS = dict(loop=372, load_gap=3, loadcnt=1, kmcnt=2, mul_u64=1, saveexec=2, vgpr=240)


class Kernel:
    def __init__(self, name):
        self.name = name
        self.ins = []           # (mnemonic, operands), program order
        self.labels = {}        # label -> instruction index
        self.vgpr = self.sgpr = self.spill = None


def parse(path):
    """Every function body in the listing, keyed by the function's own label.

    The kernel descriptor (`.amdhsa_kernel` ... `.amdhsa_next_free_vgpr`) is emitted *after* the
    body it describes, so the descriptor numbers are matched to the name they carry, not to
    whatever body was parsed last.
    """
    out, cur, desc = {}, None, None
    for line in open(path):
        s = line.split(";")[0].rstrip()
        m = LABEL.match(s)
        if m:
            lab = m.group(1)
            if lab.startswith(".L"):
                if cur is not None:
                    cur.labels[lab] = len(cur.ins)
            else:                                   # a function body starts here
                cur = out.setdefault(lab, Kernel(lab))
            continue
        if s.startswith("\t.amdhsa_kernel "):
            desc = s.split()[-1]
            out.setdefault(desc, Kernel(desc))
            continue
        if desc is not None and s.startswith("\t\t.amdhsa_"):
            key, _, val = s.strip()[len(".amdhsa_"):].partition(" ")
            attr = {"next_free_vgpr": "vgpr", "next_free_sgpr": "sgpr",
                    "private_segment_fixed_size": "spill"}.get(key)
            if attr:
                setattr(out[desc], attr, int(val))
            continue
        m = INSTR.match(s)
        if m and cur is not None and not m.group(1).startswith("."):
            cur.ins.append((m.group(1), m.group(2).strip()))
    return {k: v for k, v in out.items() if v.ins}


def is_branch(mn):
    return mn.startswith(("s_branch", "s_cbranch"))


def target(mn, ops):
    return ops.split()[0].split(",")[-1].lstrip("@").strip()


def blocks(k):
    """Basic blocks: {start: (end, [successor starts])}. Boundaries are labels and post-branch."""
    starts = sorted(set(k.labels.values()) | {0} | {i + 1 for i, (mn, _) in enumerate(k.ins)
                                                       if is_branch(mn) and i + 1 < len(k.ins)})
    owner = {}
    for n, s in enumerate(starts):
        e = (starts[n + 1] - 1) if n + 1 < len(starts) else len(k.ins) - 1
        for i in range(s, e + 1):
            owner[i] = s
    blk = {}
    for s in starts:
        e = (starts[starts.index(s) + 1] - 1) if starts.index(s) + 1 < len(starts) else len(k.ins) - 1
        mn, ops = k.ins[e]
        su = []
        if is_branch(mn):
            t = k.labels.get(target(mn, ops))
            if t is not None:
                su.append(owner[t])
            if mn.startswith("s_cbranch"):
                su.append(owner.get(e + 1))
        else:
            su.append(owner.get(e + 1))
        blk[s] = (e, [x for x in su if x is not None])
    return blk


def natural_loop(blk, head, latch):
    """The body of the back edge latch -> head.

    Two-sided on purpose: reachable from the head AND able to reach the latch. A rotated loop has
    its entry block inside the head's own region (the preheader jumps past the head), so the
    textbook "everything that reaches the latch without passing the head" would swallow the
    prologue.
    """
    succ, preds = {}, {}
    for s, (_, su) in blk.items():
        succ[s] = su
        for n in su:
            preds.setdefault(n, set()).add(s)

    def walk(seed, edges):
        seen, frontier = set(), [seed]
        while frontier:
            s = frontier.pop()
            if s in seen:
                continue
            seen.add(s)
            frontier.extend(n for n in edges.get(s, ()) if n not in seen)
        return seen

    return sorted(walk(head, succ) & walk(latch, preds))


def measure(k, blk, body):
    seg = [(i, k.ins[i]) for s in body for i in range(s, blk[s][0] + 1)]
    loads = [i for i, (mn, _) in seg if mn.startswith(("global_load", "buffer_load"))]
    blk_of = {i: s for s in body for i in range(s, blk[s][0] + 1)}
    same = len(loads) == 2 and blk_of[loads[0]] == blk_of[loads[1]]
    between = [(i, op) for i, (mn, op) in seg if mn.startswith("s_wait_loadcnt")
               and same and loads[0] < i < loads[1]]
    waits = [(i, _num(op)) for i, (mn, op) in seg if mn.startswith("s_wait_loadcnt")]
    return dict(
        loop=len(seg),
        loads=len(loads),
        load_gap=(loads[1] - loads[0]) if same else None,
        drain_between=sum(1 for _, v in between if v == 0),
        loadcnt=[v for _, v in waits],
        kmcnt=sum(1 for _, (mn, _) in seg if mn.startswith("s_wait_kmcnt")),
        mul_u64=sum(1 for _, (mn, _) in seg if mn.startswith("s_mul_u64")),
        saveexec=sum(1 for _, (mn, _) in seg if mn.startswith("s_and_saveexec")),
        execz=sum(1 for _, (mn, _) in seg if "execz" in mn),
        bar_signal=sum(1 for _, (mn, _) in seg if mn == "s_barrier_signal"),
        wmma=sum(1 for _, (mn, _) in seg if mn.startswith("v_wmma")),
        vgpr=k.vgpr, sgpr=k.sgpr, spill=k.spill,
    )


def _num(ops):
    m = re.search(r"(-?\d+)", ops)
    return int(m.group(1)) if m else None


def violations(m):
    v = []
    if m["loop"] > LIMITS["loop"]:
        v.append(f"loop body {m['loop']} > {LIMITS['loop']} instructions")
    if m["loads"] != 2:
        v.append(f"{m['loads']} global loads in the loop, expected 2")
    elif m["load_gap"] is None:
        v.append("the two global loads are not in one basic block: the K and V prefetch is not one"
                 " predicated region")
    elif m["load_gap"] > LIMITS["load_gap"]:
        v.append(f"the global loads are {m['load_gap']} instructions apart (> {LIMITS['load_gap']})")
    if m["drain_between"]:
        v.append("a full load drain (s_wait_loadcnt 0) sits between the two global loads")
    if m["loadcnt"]:
        if any(x > LIMITS["loadcnt"] for x in m["loadcnt"][:-1]):
            v.append(f"s_wait_loadcnt {m['loadcnt'][:-1]} before the tail leaves more than"
                     f" {LIMITS['loadcnt']} load in flight")
        if m["loadcnt"][-1] != 0:
            v.append(f"the tail drain is s_wait_loadcnt {m['loadcnt'][-1]}, expected 0")
    if m["kmcnt"] > LIMITS["kmcnt"]:
        v.append(f"{m['kmcnt']} s_wait_kmcnt > {LIMITS['kmcnt']}: the block-table index is awaited"
                 " in the iteration that consumes it")
    if m["mul_u64"] > LIMITS["mul_u64"]:
        v.append(f"{m['mul_u64']} s_mul_u64 > {LIMITS['mul_u64']}: the two prefetch address chains"
                 " do not share their block address")
    if m["saveexec"] > LIMITS["saveexec"]:
        v.append(f"{m['saveexec']} exec save/restore > {LIMITS['saveexec']}: the two prefetches are"
                 " not in one predicated region")
    if m["vgpr"] is not None and m["vgpr"] > LIMITS["vgpr"]:
        v.append(f"{m['vgpr']} VGPR > {LIMITS['vgpr']}: occupancy or spill")
    if m["spill"]:
        v.append(f"{m['spill']} bytes of spill")
    return v


KEYS = ("loop", "wmma", "loads", "load_gap", "drain_between", "loadcnt", "kmcnt", "mul_u64",
        "saveexec", "execz", "bar_signal", "vgpr", "sgpr", "spill")


def main(argv=None):
    ap = argparse.ArgumentParser(description="R4D prefill-attention loop schedule lint")
    ap.add_argument("asm", help="device assembly of r4d_attn_paged_h256_gqa6.hip (hipcc -S)")
    ap.add_argument("--tile", type=int, default=16,
                    help="key tile of the instance to lint (16; the thresholds are calibrated on"
                         " the tile-16 instance, the other tiles are informational)")
    ap.add_argument("--mask", type=lambda s: int(s, 0), default=None,
                    help="exact OPT mask (default: any mask with the O_DB bit)")
    ap.add_argument("--qf32", action="store_true",
                    help="lint the fp32-query instance (O_QF32) instead of the bf16 one")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.tile != 16:
        print(f"note: the thresholds are calibrated on the tile-16 instance; tile {a.tile} is"
              " informational only", file=sys.stderr)

    hits = []
    for name, k in parse(a.asm).items():
        m = KERNEL.search(name)
        if not m:
            continue
        tile, opt = int(m.group(2)), int(m.group(7))
        if tile == a.tile and opt & O_DB and bool(opt & O_QF32) == a.qf32 \
                and (a.mask is None or opt == a.mask):
            hits.append((name, k, opt))
    if len(hits) != 1:
        print(f"{a.asm}: {len(hits)} prefill instances with tile {a.tile} and the O_DB bit,"
              " expected exactly 1", file=sys.stderr)
        return 2
    name, k, opt = hits[0]

    blk = blocks(k)
    want_wmma = 2 * (int(KERNEL.search(name).group(3)) // 16)   # QK + PV per key tile
    best, seen = None, set()
    for s, (e, _) in blk.items():
        mn, ops = k.ins[e]
        if not is_branch(mn):
            continue
        t = k.labels.get(target(mn, ops))
        if t is None or t >= s:
            continue
        body = natural_loop(blk, t, s)
        if tuple(body) in seen:
            continue
        seen.add(tuple(body))
        seg = [k.ins[i] for b in body for i in range(b, blk[b][0] + 1)]
        if not any(mn.startswith("s_barrier") for mn, _ in seg):
            continue
        if sum(1 for mn, _ in seg if mn.startswith("v_wmma")) != want_wmma:
            continue
        n = sum(blk[b][0] + 1 - b for b in body)
        if best is None or n < best[0]:
            best = (n, t, body)
    if best is None:
        print(f"{name}: no loop with a barrier and {want_wmma} WMMA found", file=sys.stderr)
        return 2

    n, head, body = best
    m = measure(k, blk, body)
    bad = violations(m)
    if a.json:
        print(json.dumps(dict(instance=name, opt=opt, tile=a.tile, head=head, metrics=m,
                              violations=bad), indent=1))
    else:
        print(f"{name}\n  OPT {opt} tile {a.tile}, {len(blk)} blocks, loop head at {head},"
              f" {len(k.ins)} instructions in the kernel")
        for key in KEYS:
            print(f"  {key:16s} {m[key]}")
        print("  " + ("FAIL: " + "; ".join(bad) if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
