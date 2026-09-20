#!/usr/bin/env python3
"""GPU-less check of radiance_attn_decode: which launches it touches, which it leaves alone.

torch and aiter are stubbed (the stub `get_unified_attention_config` returns aiter's flat gfx1201 numbers,
the stub `use_2d_kernel` is aiter's own rule), so this runs anywhere: it proves the gating, the
consistency between the three ops of one launch, the 2D/3D switch and the env knobs, not the speed.

  python3 sly/check_attn_decode.py
"""
import importlib
import os
import sys
import types
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FP8, BF16 = "fp8", "bf16"
P = namedtuple("P", "q_dtype kv_cache_dtype head_size all_decode max_seqlen_q max_seqlen_k sliding_window "
                    "shuffled_kv_cache use_qq_bias use_alibi_slopes num_queries_per_kv num_tokens num_seqs "
                    "num_kv_heads num_sms num_2d_prgms target_num_prgms block_size")


def params(nseq=1, width=8, kmax=262144, q=FP8, kv=FP8, head=256, sw=0, **kw):
    bq = 16 // 6                                       # aiter's own BLOCK_Q at the stock BLOCK_M
    d = dict(q_dtype=q, kv_cache_dtype=kv, head_size=head, all_decode=width == 1, max_seqlen_q=width,
             max_seqlen_k=kmax, sliding_window=sw, shuffled_kv_cache=False, use_qq_bias=False,
             use_alibi_slopes=False, num_queries_per_kv=6, num_tokens=nseq * width, num_seqs=nseq,
             num_kv_heads=4, num_sms=32, num_2d_prgms=(nseq * width // bq + nseq) * 4, target_num_prgms=128,
             block_size=896)
    d.update(kw)
    return P(**d)


def stub_env():
    torch = types.ModuleType("torch")
    torch.float8_e4m3fn = FP8
    aiter = types.ModuleType("aiter")
    mods = {"aiter": aiter}
    for path in ("ops", "ops.triton", "ops.triton.attention", "ops.triton.utils"):
        mods["aiter." + path] = types.ModuleType("aiter." + path)
    UA = types.ModuleType("aiter.ops.triton.attention.unified_attention")
    UU = types.ModuleType("aiter.ops.triton.utils.unified_attention_utils")
    tables = {"attn_3d": dict(BLOCK_M=16, num_warps=2, num_stages=2, waves_per_eu=2),
              "kv_split": dict(TILE_SIZE=64, NUM_SEGMENTS=32),
              "reduce": dict(num_warps=2, num_stages=1, waves_per_eu=2),
              "attn_2d": dict(BLOCK_M=16, num_warps=2, num_stages=1, waves_per_eu=6, TILE_SIZE=32)}

    def get_unified_attention_config(op, params, backend="triton", arch=None):
        return dict(tables[op])

    def use_2d_kernel(params):
        return (params.sliding_window > 0 or params.max_seqlen_k <= 512
                or params.num_2d_prgms > params.target_num_prgms)

    UU.get_unified_attention_config = get_unified_attention_config
    UA.get_unified_attention_config = get_unified_attention_config
    UA.use_2d_kernel = use_2d_kernel
    UA.DEVICE_ARCH = "gfx1201"
    sys.modules.update(mods)
    sys.modules.update({"torch": torch, "aiter.ops.triton.attention.unified_attention": UA,
                        "aiter.ops.triton.utils.unified_attention_utils": UU})
    for parent, child, mod in (("aiter", "ops", mods["aiter.ops"]), ("aiter.ops", "triton", mods["aiter.ops.triton"]),
                               ("aiter.ops.triton", "attention", mods["aiter.ops.triton.attention"]),
                               ("aiter.ops.triton", "utils", mods["aiter.ops.triton.utils"])):
        setattr(sys.modules[parent], child, mod)
    mods["aiter.ops.triton.attention"].unified_attention = UA
    mods["aiter.ops.triton.utils"].unified_attention_utils = UU
    return UA, UU


def load(**env):
    for k in [k for k in os.environ if k.startswith("RADIANCE_ATTN_DECODE_")]:
        del os.environ[k]
    os.environ.update({f"RADIANCE_ATTN_DECODE_{k}": v for k, v in env.items()})
    sys.modules.pop("radiance_attn_decode", None)
    UA, UU = stub_env()
    RAD = importlib.import_module("radiance_attn_decode")
    assert RAD.install() is True
    return RAD, UA, UU


def launch(UA, UU, p):
    """What unified_attention() would end up with for p: (uses 2D?, {op: config})."""
    if UA.use_2d_kernel(p):
        return True, {}
    return False, {op: UA.get_unified_attention_config(op, p) for op in ("kv_split", "attn_3d", "reduce")}


failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("FAIL", msg)


def main():
    RAD, UA, UU = load()
    st = {op: dict(v) for op, v in {"attn_3d": dict(BLOCK_M=16, num_warps=2, num_stages=2, waves_per_eu=2),
                                     "kv_split": dict(TILE_SIZE=64, NUM_SEGMENTS=32),
                                     "reduce": dict(num_warps=2, num_stages=1, waves_per_eu=2)}.items()}

    # 1. a captured verify graph (max_model_len as max_seqlen_k) is tuned, consistently across the three ops
    for nseq in range(1, 9):
        p = params(nseq=nseq)
        is2d, cfg = launch(UA, UU, p)
        check(not is2d, f"nseq={nseq} width 8 at capture must take the 3D kernel")
        if is2d:
            continue
        plan = RAD.plan(p)
        check(plan is not None, f"nseq={nseq}: no plan")
        check(cfg["kv_split"]["TILE_SIZE"] == plan["tile"], f"nseq={nseq}: kv_split tile != plan tile")
        seg = cfg["kv_split"]["NUM_SEGMENTS"]
        check(seg == plan["segments"] and seg & (seg - 1) == 0 and RAD.SPLIT_MIN <= seg <= RAD.SPLIT_MAX,
              f"nseq={nseq}: bad segments {seg}")
        check(cfg["attn_3d"]["BLOCK_M"] == plan["block_m"], f"nseq={nseq}: BLOCK_M mismatch")
        check(cfg["reduce"]["num_warps"] == plan["r_warps"], f"nseq={nseq}: reduce warps mismatch")
        # the grid must be the plan's own launch and fit the machine
        check(plan["prgms"] == (p.num_tokens // max(1, plan["block_m"] // 6) + nseq) * 4, f"nseq={nseq}: prgms")
        print(f"nseq={nseq}: BLOCK_M={plan['block_m']} tile={plan['tile']} splits={seg} prgms={plan['prgms']} "
              f"warps={plan['warps']}/{plan['r_warps']}")

    # 2. everything outside the regime is left exactly as aiter has it
    untouched = {
        "all_decode (width 1)": params(width=1),
        "bf16 kv": params(kv=BF16),
        "bf16 q": params(q=BF16),
        "head 128": params(head=128),
        "sliding window": params(sw=128),
        "shuffled kv": params(shuffled_kv_cache=True),
    }
    for name, p in untouched.items():
        check(RAD.plan(p) is None, f"{name}: plan() must be None")
        for op in ("kv_split", "attn_3d", "reduce"):
            check(UA.get_unified_attention_config(op, p) == st[op], f"{name}: {op} config was changed")
        check(UA.use_2d_kernel(p) == (p.sliding_window > 0 or p.max_seqlen_k <= 512 or p.num_2d_prgms > 128),
              f"{name}: 2D/3D choice was changed")

    # 3. short contexts keep the 2D kernel; an eager call sizes its splits from the real depth
    check(UA.use_2d_kernel(params(kmax=400)), "max_seqlen_k <= 512 must stay on the 2D kernel")
    seg_small = RAD.plan(params(kmax=2048))["segments"]
    seg_big = RAD.plan(params(kmax=262144))["segments"]
    check(seg_small <= seg_big, f"splits must not grow when the depth shrinks ({seg_small} vs {seg_big})")

    # 4. width gate (measured crossover ~50% fill): 6..10 tokens (36..60 of 64 rows) take the 64-row block,
    #    2..5 the 32-row one; BLOCK_M 16 is never picked
    for w, bm in ((2, 32), (3, 32), (5, 32), (6, 64), (7, 64), (8, 64), (9, 64), (10, 64)):
        check(RAD.plan(params(width=w))["block_m"] == bm, f"width {w}: BLOCK_M must be {bm}")
    for w in (11, 16, 256, 2048):             # 66+ rows: prefill chunks, left to aiter (2D kernel included)
        check(RAD.plan(params(width=w)) is None, f"width {w}: plan() must be None")
        check(UA.get_unified_attention_config("kv_split", params(width=w)) == st["kv_split"], f"width {w}: config changed")
    # 4b. split count: 32 with one sequence (4 programs), 16 beyond, never past the depth in an eager call
    check(RAD.plan(params(nseq=1))["segments"] == 32, "1 sequence: 32 splits")
    for nseq in range(2, 9):
        check(RAD.plan(params(nseq=nseq))["segments"] == 16, f"{nseq} sequences: 16 splits")
    check(RAD.plan(params(kmax=600))["segments"] <= 32, "eager call at 600 tokens")

    # 5. knobs
    RAD, UA, UU = load(TUNE="0")
    check(RAD.plan(params()) is None, "TUNE=0 must disable the plan")
    check(not getattr(UU.get_unified_attention_config, "_radiance_decode_tune", False), "TUNE=0 must not wrap")
    RAD, UA, UU = load(WIDE="0")
    check(RAD.plan(params())["block_m"] == 32, "WIDE=0 must never widen (32-row cell)")
    RAD, UA, UU = load(WIDE="1")
    check(RAD.plan(params(width=5))["block_m"] == 64, "WIDE=1 must widen a 5-token verify")
    RAD, UA, UU = load(MIN_FILL="0.8")
    check(RAD.plan(params(width=8))["block_m"] == 32, "MIN_FILL=0.8 must not widen a 48-row verify")
    RAD, UA, UU = load(**{"3D": "0"})
    check(UA.use_2d_kernel(params(nseq=8)), "3D=0 must keep aiter's 2D choice at 8 sequences")

    # 6. idempotent, and a module that had already bound the stock function is rebound
    RAD, UA, UU = load()
    wrapped = UU.get_unified_attention_config
    check(RAD.install() is True and UU.get_unified_attention_config is wrapped, "install() is not idempotent")
    check(UA.get_unified_attention_config is wrapped, "unified_attention's own binding was not rebound")

    # 7. a failure inside the tune falls back to aiter's numbers instead of raising
    RAD, UA, UU = load()
    RAD._plan = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    check(UA.get_unified_attention_config("kv_split", params()) == st["kv_split"], "fallback did not return aiter's config")

    if failures:
        print(f"FAIL: {len(failures)} finding(s)")
        sys.exit(1)
    print("radiance_attn_decode check OK")


if __name__ == "__main__":
    main()
