#!/usr/bin/env python3
"""GPU gate for the A1 option bits of the R4D prefill attention kernel (O_CFOLD, O_MAX3).

[GPU window only] Not a CI test: it needs the R9700 and a libr4d .so built from a tree that carries
this patch. Run it on the gate unit, one process per KV leg (R4D_ATTN_FP8 is cached per process;
R4D_ATTN_OPT2 is not, so the arms can be armed in-process ABBA):

    R4D_ATTN_FP8=3 python3 sly/r4d/tests/gpu_attn_prefill_opt2.py installed 0,16356,40890 opts_fp8_3.json
    R4D_ATTN_FP8=0 python3 sly/r4d/tests/gpu_attn_prefill_opt2.py installed 0,16356,40890 opts_fp8_0.json

Arms = R4D_ATTN_OPT2 in {0 base, 3 O_MAX3 alone, 2 O_CFOLD alone, 1 both}. Per (ctx, arm):

  * O_MAX3 alone must be BIT-IDENTICAL to the base arm (sha256 over the output bytes). The kernel
    only reorders fmax, and LLVM already emits v_max3_num_f32 for the base chain, so this arm also
    has to come out at the same VGPR count -- if it does not, the control is void.
  * every arm's row relRMSE against the fp32 reference (the chunked form of
    sly/r4d/tests/test_attn_prefill_ref.py `prefill_reference`) must not exceed 1.05 x the base
    arm's, and no arm may emit NaN.
  * timing: 5 warmups, then 20 (ctx < 20k) or 10 reps, median, arms back to back and the base arm
    again last (ABBA), sclk sampled alongside.

GO criterion (A1, decided on this output): both-arm >= 4 % faster than base at ctx 16356 AND 40890.
The script prints the verdict; it does not fail on it, because "slower" is a legitimate outcome.
"""
import faulthandler
import glob
import hashlib
import json
import os
import sys
import threading
import time

faulthandler.enable()

import torch                                                   # the gate runs where torch+hip exist

D, H, KVH, GQA, BS, M = 256, 24, 4, 6, 16, 4089
SCALE = D ** -0.5
ARMS = [(0, "base"), (3, "max3"), (2, "cfold"), (1, "both")]


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


def unpack(kv, bt, ctx, kvh, dev):
    """[nblk, kvh, BS, 2D] fp8 e4m3 + block table -> [ctx, D] float K and V, block order undone."""
    pos = torch.arange(ctx, device=dev)
    blk = bt[0][pos // BS].long()
    x = kv[blk, kvh, pos % BS].float()
    return x[:, :D], x[:, D:]


def reference(q, kv, bt, ctx, kvh, dev, chunk=128):
    """fp32 oracle, the same math as the CPU test's prefill_reference, chunked over q rows."""
    K, V = unpack(kv, bt, ctx, kvh, dev)
    out = torch.empty(M, GQA, D, device=dev)
    lim = torch.arange(M, device=dev) + (ctx - M)
    keys = torch.arange(ctx, device=dev).unsqueeze(0)
    for h in range(GQA):
        qh = q[:, kvh * GQA + h].float() * SCALE                       # FOLDQ
        for a in range(0, M, chunk):
            s = qh[a:a + chunk] @ K.T
            s = s.masked_fill(keys > lim[a:a + chunk].unsqueeze(1), -float("inf"))
            out[a:a + chunk, h] = torch.softmax(s, dim=-1) @ V
    return out


def row_relrmse(o, ref):
    num = (o.float() - ref).norm(dim=-1)
    den = ref.norm(dim=-1).clamp_min(1e-30)
    r = (num / den)
    return float(r.median()), float(r.max())


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    so, ctxs = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    out_json = sys.argv[3]
    leg = int(os.environ.get("R4D_ATTN_FP8", "0"))
    os.environ.setdefault("R4D_ATTN_FP8", str(leg))

    if so != "installed":
        sys.path.insert(0, so)
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)
    import r4d                                                     # noqa: E402
    print("r4d from", r4d.__file__, "| leg FP8 =", leg, flush=True)
    name = r4d.select("attn_prefill_paged", head_dim=D, gqa=GQA, block_size=BS, causal=1,
                      q_dtype="bf16", kv_dtype="fp8_e4m3")
    launch = getattr(r4d, name)

    dev = "cuda"
    torch.manual_seed(0)
    q = (torch.randn(M, H, D, device=dev) * 0.5).to(torch.bfloat16)
    out = torch.empty(M, H, D, device=dev, dtype=torch.bfloat16)
    kd = torch.ones(KVH, device=dev, dtype=torch.float32)
    vd = torch.ones(KVH, device=dev, dtype=torch.float32)
    scratch = torch.empty(256, device=dev, dtype=torch.uint8)
    stream = torch.cuda.current_stream().cuda_stream
    res, verdicts = [], []

    for c in ctxs:
        ctx = c + M
        nblk = (ctx + BS - 1) // BS
        g = torch.Generator(device="cpu").manual_seed(1)
        kvf = torch.randn(nblk, KVH, BS, 2 * D, generator=g).to(torch.float8_e4m3fn).to(dev)
        kv = kvf.view(torch.uint8)                                  # the kernel takes raw bytes
        bt = torch.randperm(nblk, generator=g).to(torch.int32).to(dev).view(1, -1)
        sl = torch.tensor([ctx], dtype=torch.int32, device=dev)
        ref = reference(q, kvf, bt, ctx, 0, dev)

        def run(opt2):
            os.environ["R4D_ATTN_OPT2"] = str(opt2)
            launch(q.data_ptr(), kv.data_ptr(), bt.data_ptr(), sl.data_ptr(), out.data_ptr(),
                   kd.data_ptr(), vd.data_ptr(), scratch.data_ptr(), 1, M, H, KVH, D, BS, nblk,
                   KVH * BS * 2 * D, BS * 2 * D, SCALE, 0, ctx, stream)

        arms = {}
        for opt2, label in ARMS:                                   # correctness in one pass each
            run(opt2)
            torch.cuda.synchronize()
            o = out.clone()
            arms[label] = dict(opt2=opt2,
                               digest=hashlib.sha256(o.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16],
                               nan=bool(torch.isnan(o.float()).any()),
                               relrmse=row_relrmse(o, ref))

        # ABBA: base, the variants, base again. Take the better of the two base medians as baseline.
        ms = {}
        for opt2, label in [(0, "base"), (2, "cfold"), (1, "both"), (3, "max3"), (0, "base")]:
            run(opt2)
            torch.cuda.synchronize()
            for _ in range(5):
                run(opt2)
            torch.cuda.synchronize()
            n = 20 if c < 20000 else 10
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            s = Sampler()
            s.start()
            ts = []
            for _ in range(n):
                e0.record()
                run(opt2)
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1))
            s.stop = True
            s.join()
            ts.sort()
            key = label if label not in ms else label + "_2"
            clk = sorted(x["sclk"] for x in s.samples if "sclk" in x)
            ms[key] = dict(ms_med=ts[len(ts) // 2], ms_min=ts[0],
                           sclk=clk[len(clk) // 2] if clk else None)

        base_ms = min(ms["base"]["ms_med"], ms["base_2"]["ms_med"])
        for label in ("max3", "cfold", "both"):
            arms[label]["speedup"] = base_ms / ms[label]["ms_med"]
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
        res.append(dict(ctx=c, leg=leg, arms=arms, time=ms, checks=ok or "ok"))
        print(json.dumps(res[-1], default=float), flush=True)
        del kv, kvf, bt, ref
        torch.cuda.empty_cache()

    print("GO" if all(v >= 1.04 for v in verdicts) and verdicts else "NO-GO", 
          f"(both-arm speedup at the long ctxs: {['%.3f' % v for v in verdicts]})", flush=True)
    with open(out_json, "a") as f:
        json.dump(res, f)
        f.write("\n")
    return 0 if all(r["checks"] == "ok" for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
