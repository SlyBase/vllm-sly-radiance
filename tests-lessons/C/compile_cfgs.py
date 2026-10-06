# GPU-less compile of the lm_head W4A16 kernels for gfx1201: vgpr / spill / scratch per config
import sys, itertools, re
import triton, triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as h
tgt = GPUTarget("hip", "gfx1201", 32)
def stats(co):
    s = co.asm["amdgcn"]
    g = lambda p: int((re.search(p, s) or [0, -1])[1])
    return g(r"\.vgpr_count:\s+(\d+)"), g(r"\.vgpr_spill_count:\s+(\d+)"), g(r"\.private_segment_fixed_size:\s+(\d+)")
def stock(bm, bn, bk, w, st):
    sig = {"a_ptr": "*bf16", "b_ptr": "*i32", "scales_ptr": "*bf16", "zp_ptr": "*bf16", "c_ptr": "*bf16",
           "M": "i32", "N": "i32", "K": "i32", "K8": "i32", "num_groups": "i32", "group_size": "i32"}
    const = {"ZP_BIAS": 8, "HAS_ZP": False, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "LAYOUT": 1}
    src = ASTSource(h._triton_w4a16_skinny_fmt_kernel, sig, const)
    opts = {"num_warps": w}
    if st is not None: opts["num_stages"] = st
    return triton.compile(src, target=tgt, options=opts)
def splitk(bm, bn, bk, w, st, deq, unpack, kstep, mode=3):
    sig = {"a_ptr": "*bf16", "b_ptr": "*i32", "scales_ptr": "*bf16", "zp_ptr": "*bf16", "p_ptr": "*fp32", "c_ptr": "*bf16",
           "c2_ptr": "*bf16", "lock_ptr": "*i32", "M": "i32", "N": "i32", "K": "i32", "K8": "i32", "num_groups": "i32",
           "tiles_per_split": "i32", "n_split": "i32", "N1": "i32", "group_size": "i32"}
    const = {"ZP_BIAS": 8, "HAS_ZP": False, "MODE": mode, "EPI": 0, "DEQ": deq, "UNPACK": unpack, "LAYOUT": 1,
             "KSTEP": kstep, "MAGIC": 0x4300, "MAGIC_F": 128.0, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}
    src = ASTSource(h._radiance_w4a16_splitk_kernel, sig, const)
    opts = {"num_warps": w}
    if st is not None: opts["num_stages"] = st
    return triton.compile(src, target=tgt, options=opts)
def check_cands():
    # --cands: re-check the microbench candidates (cands.py)
    from cands import CANDS
    bad = 0
    for b, lst in sorted(CANDS.items()):
        for cfg in lst:
            bm, bn, bk, w, st, _sk, _mode, deq, unpack = cfg[:9]
            v, sp, pv = stats(splitk(bm, bn, bk, w, st, deq, unpack, 1))
            bad += sp > 0
            print(f"M<={b:2d} {cfg} vgpr={v} spill={sp} scratch={pv}B" + ("  <-- SPILLS" if sp else ""), flush=True)
    print("candidates with spills:", bad)


if __name__ == "__main__" and "--cands" in sys.argv:
    check_cands()
elif __name__ == "__main__":
    print("== production table (stock kernel, tiled) ==")
    for M in (8, 16, 32, 40, 64):
        bm, bn, bk, w, st = h._GFX12X_DRAFT_OVERRIDES[(128, 5120, 248320, M)]
        v, sp, pv = stats(stock(bm, bn, bk, w, st))
        print(f"M<={M:2d} cfg={(bm,bn,bk,w,st)} vgpr={v} spill={sp} scratch={pv}B")
    print("== lean candidates (split-K kernel, split 1, MODE 3) ==")
    for M, cfgs in {32: [(32,128,128,4,None),(32,64,128,4,None),(32,128,128,8,None)],
                    40: [(64,64,128,4,None),(64,128,128,8,None),(64,128,64,8,None),(64,64,64,4,None)],
                    64: [(64,64,128,4,None),(64,128,128,8,None),(64,128,64,8,None),(64,64,64,4,None)]}.items():
        for (bm,bn,bk,w,st) in cfgs:
            for deq, unpack, kstep in ((0,0,1),(1,0,1),(2,0,1),(1,1,1),(2,1,1),(1,0,2)):
                if unpack and bk < 128: continue
                try:
                    v, sp, pv = stats(splitk(bm,bn,bk,w,st,deq,unpack,kstep))
                    print(f"M{M:2d} {(bm,bn,bk,w,st)} deq={deq} unpack={unpack} kstep={kstep} vgpr={v} spill={sp} scratch={pv}B", flush=True)
                except Exception as e:
                    print(f"M{M:2d} {(bm,bn,bk,w,st)} deq={deq} unpack={unpack} kstep={kstep} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
