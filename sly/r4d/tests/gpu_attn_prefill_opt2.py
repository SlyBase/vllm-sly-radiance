#!/usr/bin/env python3
"""GPU gate for the A1 option bits of the R4D prefill attention kernel (O_CFOLD, O_MAX3).

[GPU window only] Not a CI test: it needs the R9700 and a libr4d .so built from a tree that carries
this patch. Run it on the gate unit, one process per KV leg (R4D_ATTN_FP8 is `static` in the
launcher, so a process can only ever measure one leg):

    R4D_ATTN_FP8=3 python3 sly/r4d/tests/gpu_attn_prefill_opt2.py installed 0,16356,40890 opts_fp8_3.json
    R4D_ATTN_FP8=0 python3 sly/r4d/tests/gpu_attn_prefill_opt2.py installed 0,16356,40890 opts_fp8_0.json

R4D_ATTN_OPT2 is read once per process too (`static`, exactly like R4D_ATTN_FP8) and only arms the
fp8mode == 3 leg -- on the legs that already spill it moves the spill the wrong way. So the arms cannot
be armed in-process: this script spawns one child per arm (`--arm OPT2 SO CTXS OUT`) and walks
base, cfold, both, max3, base again (ABBA over processes). Arms = R4D_ATTN_OPT2 in
{0 base, 2 O_CFOLD alone, 1 both, 3 O_MAX3 alone}. Per (ctx, arm):

  * O_MAX3 alone must be BIT-IDENTICAL to the base arm (sha256 over the output bytes). The kernel
    only reorders fmax, and LLVM already emits v_max3_num_f32 for the base chain, so the arm is a
    codegen no-op by construction -- it is the control, not an optimisation.
  * every arm's row relRMSE against the fp32 reference of sly/r4d/tests/test_attn_prefill_ref.py
    (`prefill_reference_all`, i.e. the same oracle the CPU test validates against SDPA) must not
    exceed 1.05 x the base arm's, and no arm may emit NaN.
  * timing: 5 warmups, then 20 (ctx < 20k) or 10 reps, median, sclk sampled alongside.

GO criterion (A1, decided on this output): both-arm >= 4 % faster than base at ctx 16356 AND 40890.
The script prints the verdict; it does not fail on it, because "slower" is a legitimate outcome.
"""
import faulthandler
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

faulthandler.enable()

import torch                                                   # the gate runs where torch+hip exist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_attn_prefill_ref import prefill_reference_all        # noqa: E402  one oracle, not two

D, H, KVH, GQA, BS, M = 256, 24, 4, 6, 16, 4089
SCALE = D ** -0.5
ARMS = [(0, "base"), (2, "cfold"), (1, "both"), (3, "max3")]
# ABBA over processes: the base arm opens and closes the series so drift shows up in base/base_2.
ORDER = [(0, "base"), (2, "cfold"), (1, "both"), (3, "max3"), (0, "base_2")]


def hw():
    r = {}
    for p in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/power1_average"):
        try:
            r["W"] = int(open(p).read()) / 1e6
        except OSError:
            pass
    for p in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/freq1_input"):
        try:
            r["sclk"] = int(open(p).read()) / 1e6
        except OSError:
            pass
    return r


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], False

    def run(self):
        while not self.stop:
            self.samples.append(hw())
            time.sleep(0.05)


def reference(q, kv, bt, ctx, kvheads, chunk=128):
    """fp32 oracle over ALL kv heads, [M, kvheads*GQA, D] -- the shape the launcher fills.

    Reuses `prefill_reference_all` from the CPU test (`kvh*GQA + hi` is served by kv head `kvh`, the
    kernel's grid.y), so the oracle the gate judges against is the one validated against
    scaled_dot_product_attention there.
    """
    return prefill_reference_all(q, kv, bt, ctx, kvheads, 1.0, 1.0, chunk=chunk)


def row_relrmse(o, ref):
    num = (o.float() - ref).norm(dim=-1)
    den = ref.norm(dim=-1).clamp_min(1e-30)
    r = (num / den)
    return float(r.median()), float(r.max())


def load_launch(so):
    if so != "installed":
        sys.path.insert(0, so)
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)
    import r4d                                                     # noqa: E402
    print("r4d from", r4d.__file__, "| leg FP8 =", os.environ.get("R4D_ATTN_FP8", "0"),
          "| arm OPT2 =", os.environ.get("R4D_ATTN_OPT2", "0"), flush=True)
    name = r4d.select("attn_prefill_paged", head_dim=D, gqa=GQA, block_size=BS, causal=1,
                      q_dtype="bf16", kv_dtype="fp8_e4m3")
    return getattr(r4d, name)


def arm_main(opt2, so, ctxs, out_json):
    """One arm in one process: the launcher caches R4D_ATTN_OPT2, so this is the only way to A/B it."""
    os.environ["R4D_ATTN_OPT2"] = str(opt2)
    launch = load_launch(so)
    leg = int(os.environ.get("R4D_ATTN_FP8", "0"))

    dev = "cuda"
    torch.manual_seed(0)
    res = []
    for c in ctxs:
        ctx = c + M
        nblk = (ctx + BS - 1) // BS
        g = torch.Generator(device="cpu").manual_seed(1)
        q = (torch.randn(M, H, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)
        kvf = torch.randn(nblk, KVH, BS, 2 * D, generator=g).to(torch.float8_e4m3fn).to(dev)
        kv = kvf.view(torch.uint8)                                  # the kernel takes raw bytes
        bt = torch.randperm(nblk, generator=g).to(torch.int32).to(dev).view(1, -1)
        out = torch.empty(M, H, D, device=dev, dtype=torch.bfloat16)
        kd = torch.ones(KVH, device=dev, dtype=torch.float32)
        vd = torch.ones(KVH, device=dev, dtype=torch.float32)
        scratch = torch.empty(256, device=dev, dtype=torch.uint8)
        stream = torch.cuda.current_stream().cuda_stream
        sl = torch.tensor([ctx], dtype=torch.int32, device=dev)

        def run():
            launch(q.data_ptr(), kv.data_ptr(), bt.data_ptr(), sl.data_ptr(), out.data_ptr(),
                   kd.data_ptr(), vd.data_ptr(), scratch.data_ptr(), 1, M, H, KVH, D, BS, nblk,
                   KVH * BS * 2 * D, BS * 2 * D, SCALE, 0, ctx, stream)

        ref = reference(q, kvf, bt, ctx, KVH)
        run()
        torch.cuda.synchronize()
        o = out.clone()
        digest = hashlib.sha256(o.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16]
        nan = bool(torch.isnan(o.float()).any())
        relrmse = row_relrmse(o, ref)

        for _ in range(5):
            run()
        torch.cuda.synchronize()
        n = 20 if c < 20000 else 10
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        s = Sampler()
        s.start()
        ts = []
        for _ in range(n):
            e0.record()
            run()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        s.stop = True
        s.join()
        ts.sort()
        clk = sorted(x["sclk"] for x in s.samples if "sclk" in x)
        res.append(dict(ctx=c, leg=leg, opt2=opt2, digest=digest, nan=nan, relrmse=relrmse,
                        ms_med=ts[len(ts) // 2], ms_min=ts[0],
                        sclk=clk[len(clk) // 2] if clk else None))
        print(json.dumps(res[-1], default=float), flush=True)
        del kv, kvf, bt, ref, o
        torch.cuda.empty_cache()
    with open(out_json, "w") as f:
        json.dump(res, f)
    return 0


def gate_main(so, ctxs, out_json):
    """Parent: one child per arm, then the A1 decision on the merged numbers."""
    leg = int(os.environ.get("R4D_ATTN_FP8", "0"))
    per_arm = {}
    for opt2, label in ORDER:
        fd, tmp = tempfile.mkstemp(prefix=f"attn_opt2_{opt2}_", suffix=".json")
        os.close(fd)
        env = dict(os.environ, R4D_ATTN_FP8=str(leg), R4D_ATTN_OPT2=str(opt2))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--arm", str(opt2), so,
                            ",".join(str(c) for c in ctxs), tmp], env=env)
        if r.returncode != 0:
            print(f"FAIL arm {label} (OPT2={opt2}) exited {r.returncode}", flush=True)
            return 1
        with open(tmp) as f:
            per_arm[label] = {d["ctx"]: d for d in json.load(f)}
        os.unlink(tmp)

    res, verdicts = [], []
    for c in ctxs:
        arms = {label: per_arm[label][c] for _, label in ARMS}
        time = {label: per_arm[label][c] for _, label in ORDER}
        base_ms = min(time["base"]["ms_med"], time["base_2"]["ms_med"])
        for label in ("max3", "cfold", "both"):
            arms[label]["speedup"] = base_ms / time[label]["ms_med"]
        ok = []
        if arms["max3"]["digest"] != arms["base"]["digest"]:
            ok.append("FAIL max3 digest != base (control is not bit-identical)")
        for label in ("base", "max3", "cfold", "both"):
            a = arms[label]
            if a["nan"]:
                ok.append(f"FAIL {label}: NaN in the output")
            if a["relrmse"][0] > 1.05 * arms["base"]["relrmse"][0] and label != "base":
                ok.append(f"FAIL {label}: row relRMSE {a['relrmse'][0]:.3e} > 1.05 x base "
                          f"{arms['base']['relrmse'][0]:.3e}")
        if c >= 15000:                                             # the GO criterion's two ctxs
            verdicts.append(arms["both"].get("speedup", 0.0))
        res.append(dict(ctx=c, leg=leg, arms=arms,
                        time={k: dict(ms_med=v["ms_med"], ms_min=v["ms_min"], sclk=v["sclk"])
                              for k, v in time.items()},
                        base_drift=time["base_2"]["ms_med"] / time["base"]["ms_med"],
                        checks=ok or "ok"))
        print(json.dumps(res[-1], default=float), flush=True)

    print("GO" if all(v >= 1.04 for v in verdicts) and verdicts else "NO-GO",
          f"(both-arm speedup at the long ctxs: {['%.3f' % v for v in verdicts]})", flush=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)                                # one valid JSON document
    return 0 if all(r["checks"] == "ok" for r in res) else 1


def main():
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == "--arm":                             # child: --arm OPT2 SO CTXS OUT
        return arm_main(int(a[1]), a[2], [int(x) for x in a[3].split(",")], a[4])
    if len(a) != 3:
        sys.exit(__doc__)
    return gate_main(a[0], [int(x) for x in a[1].split(",")], a[2])


if __name__ == "__main__":
    sys.exit(main())
