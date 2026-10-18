"""A-tiled GEMM weight-path knobs (SKIPPAD, PSW) vs production, checkpoint weights + real-producer A.

Variant v = psw + 4 * skip_pad (psw in {0, 1, 2}: set_at_psw; skip_pad: set_at_skip_pad); v = 0 is
the production launch. PSW variants get their weights repacked here with radiance_mxfp4.psw_pack_w /
psw_fold_ws (the SAME logical weights). A for gate_up/qkvz = add_rms_quant(token embeddings of real
text), A for down = silu_mul_quant(real gate_up output of the same layer).
Correctness: DIGEST (sha256 of the bf16 output) of every variant must equal v=0's, on launch_at and,
with PBCHECK=1, on launch_at_p with two A copies split at pb1 = N/2 (merged-linear partitions).
v=0 is checked against the fp32 reference (rel <= 0.02).
Timing: ABBA pairs (A = v 0, B = variant), each leg a SECS-long loop; sclk / board power sampled
from hwmon during each leg.
env: VARS (4,1,2,6), MS, SHAPES (gate_up,down,qkvz), SECS (2.5), PAIRS (2), CHECKONLY, PBCHECK, LAYERS.
"""
import glob, hashlib, json, os, sys, threading, time
import torch
sys.path.insert(0, "/w")
import bench_prefill_cells as B
import radiance_mxfp4 as R
import radiance_fused_norm  # noqa: F401
from safetensors import safe_open

dev = torch.device("cuda"); stream = torch.cuda.current_stream().cuda_stream
HW = glob.glob("/sys/class/drm/card0/device/hwmon/hwmon*")[0]
VARS = [int(x) for x in os.environ.get("VARS", "4,1,2,6").split(",") if x]
MS = [int(x) for x in os.environ.get("MS", "880,1760").split(",")]
SHAPES = os.environ.get("SHAPES", "gate_up,down,qkvz").split(",")
SECS = float(os.environ.get("SECS", "2.5"))
PAIRS = int(os.environ.get("PAIRS", "2"))
CHECKONLY = os.environ.get("CHECKONLY", "0") == "1"
PBCHECK = os.environ.get("PBCHECK", "0") == "1"
LAYERS = [int(x) for x in os.environ.get("LAYERS", "12,45,52,20,38,13").split(",")]
CK = "/ck"; PFX = "model.language_model."
IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
assert R.WPERM, "PSW is WPERM-only"
E = R._ext

samples = []; stop = [False]
def rd(f):
    try: return int(open(f"{HW}/{f}").read())
    except Exception: return -1
def sampler():
    while not stop[0]:
        samples.append((time.time(), rd("freq1_input") / 1e6, rd("power1_average") / 1e6)); time.sleep(0.1)
threading.Thread(target=sampler, daemon=True).start()


def setv(v):
    E.set_at_psw(v & 3); E.set_at_skip_pad(1 if v & 4 else 0)


def tens(key):
    with safe_open(f"{CK}/{IDX[key]}", "pt", device="cpu") as f:
        return f.get_tensor(key)


SH = {"gate_up": (34816, 5120, "mlp", ["gate_proj", "up_proj"]),
      "down": (5120, 17408, "mlp", ["down_proj"]),
      "qkvz": (16384, 5120, "linear_attn", ["in_proj_qkv", "in_proj_z"])}


def real_w(name, L):
    _, _, blk, parts = SH[name]
    w = torch.cat([tens(f"{PFX}layers.{L}.{blk}.{n}.weight_packed") for n in parts], 0).to(dev)
    s = torch.cat([tens(f"{PFX}layers.{L}.{blk}.{n}.weight_scale") for n in parts], 0).to(dev)
    N, K = w.shape[0], w.shape[1] * 2
    wp = R.permute_w(w, N, K).contiguous(); st = s.T.contiguous()
    return wp, st, R.make_row_ref(st)


_tok = [None]
def embed_input(M):
    emb = tens(f"{PFX}embed_tokens.weight")
    if _tok[0] is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CK)
        txt = "".join(open(f).read() for f in sorted(glob.glob("/opt/vllm/lib/python3.12/site-packages/vllm/*.py"))[:40])
        _tok[0] = tok(txt)["input_ids"]
    ids = (_tok[0] * (M // max(1, len(_tok[0])) + 1))[:M]
    return emb[torch.tensor(ids)].to(dev, torch.bfloat16)


def run(f, copies, secs):
    for i in range(10): f(i % copies)
    torch.cuda.synchronize()
    t0 = time.time(); ts = []; n = 0
    while time.time() - t0 < secs:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        for i in range(20): f((n + i) % copies)
        ev[1].record(); torch.cuda.synchronize(); n += 20
        ts.append(ev[0].elapsed_time(ev[1]) * 1000 / 20)
    t1 = time.time()
    ss = [x for x in samples if t0 + 0.8 <= x[0] <= t1]
    sc = sum(x[1] for x in ss) / max(1, len(ss)); pw = sum(x[2] for x in ss) / max(1, len(ss))
    return sorted(ts)[len(ts) // 2], sc, pw


def h(t):
    return hashlib.sha256(t.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


print("RES shape M var pair legA_us legB1_us legB2_us legA2_us ratio_B/A TFs_A TFs_B sclk_A sclk_B pow_A pow_B", flush=True)
for name in SHAPES:
    N, K, _, _ = SH[name]
    wbytes = N * K // 2 + N * K // 32
    copies = max(2, -(-256 * (1 << 20) // wbytes))
    layers = (LAYERS * 4)[:copies]
    real = [real_w(name, L) for L in layers]
    RF = [r[2] for r in real]
    Wv = {0: [r[0] for r in real]}; Sv = {0: [r[1] for r in real]}
    for p_ in sorted({v & 3 for v in VARS} - {0}):
        Wv[p_] = [R.psw_pack_w(w, p_) for w in Wv[0]]
    if any(v & 3 for v in VARS):
        Sv[1] = [R.psw_fold_ws(s, r) for s, r in zip(Sv[0], RF)]
    L0 = layers[0]
    gu_w = real_w("gate_up", L0) if name == "down" else None
    for M in MS:
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        with B.force_tiled():
            x = embed_input(M)
            lw = tens(f"{PFX}layers.{L0}.post_attention_layernorm.weight").to(dev, torch.bfloat16)
            lw2 = tens(f"{PFX}layers.{L0}.input_layernorm.weight").to(dev, torch.bfloat16)
            qa, sa = torch.ops.radiance.add_rms_quant(x, torch.zeros_like(x), lw, 1e-6)[:2]
            qb, sb = torch.ops.radiance.add_rms_quant(x, torch.zeros_like(x), lw2, 1e-6)[:2]
            if name != "down":
                q, s, q2, s2 = qa, sa, qb, sb
            else:
                gu = torch.empty(M, 34816, device=dev, dtype=torch.bfloat16)
                setv(0)
                E.launch_at(qa.data_ptr(), gu_w[0].data_ptr(), gu_w[1].data_ptr(), gu_w[2].data_ptr(),
                            sa.data_ptr(), gu.data_ptr(), M, 34816, 5120, stream)
                torch.cuda.synchronize()
                q, s = torch.ops.radiance.silu_mul_quant(gu)
                E.launch_at(qb.data_ptr(), gu_w[0].data_ptr(), gu_w[1].data_ptr(), gu_w[2].data_ptr(),
                            sb.data_ptr(), gu.data_ptr(), M, 34816, 5120, stream)
                torch.cuda.synchronize()
                q2, s2 = torch.ops.radiance.silu_mul_quant(gu)
        SZ = (M + 15) // 16 * 16 * K

        def mk(v):
            W = Wv[v & 3]; S = Sv[1] if v & 3 else Sv[0]
            def f(i):
                E.launch_at(q.data_ptr(), W[i].data_ptr(), S[i].data_ptr(), RF[i].data_ptr(),
                            s.data_ptr(), out.data_ptr(), M, N, K, stream)
            return f

        def digest(v):
            setv(v); mk(v)(0); torch.cuda.synchronize(); return h(out)

        if PBCHECK:
            st1 = torch.empty(0, dtype=torch.uint8, device=dev).set_(q.untyped_storage(), 0, (SZ,))
            st2 = torch.empty(0, dtype=torch.uint8, device=dev).set_(q2.untyped_storage(), 0, (SZ,))
            A2 = torch.cat([st1, st2]).contiguous()
            S2 = torch.stack([s.reshape(-1)[:M].float(), s2.reshape(-1)[:M].float()]).contiguous()
            pb1 = N // 2 // 128 * 128

            def digest_p(v):
                setv(v); W = Wv[v & 3][0]; S = (Sv[1] if v & 3 else Sv[0])[0]
                E.launch_at_p(A2.data_ptr(), W.data_ptr(), S.data_ptr(), RF[0].data_ptr(), S2.data_ptr(),
                              out.data_ptr(), M, N, K, pb1, 1 << 30, SZ, stream)
                torch.cuda.synchronize(); return h(out)
            p0 = digest_p(0)
            print(f"PDIGEST {name} M={M} pb1={pb1} var=0 {p0}", flush=True)
            for v in VARS:
                pv = digest_p(v)
                print(f"PDIGEST {name} M={M} var={v} {pv} {'SAME' if pv == p0 else 'WRONG'}", flush=True)
            del A2, S2

        d0 = digest(0)
        from tile_layout import untile
        st_ = torch.empty(0, dtype=torch.uint8, device=dev).set_(q.untyped_storage(), 0, (SZ,))
        qrm = untile(st_, M, K).view(torch.float8_e4m3fn)
        rows = min(M, 256)
        ref = R._exact_ref(qrm[:rows], s[:rows], Wv[0][0], Sv[0][0], N, K)
        rel = ((out[:rows].float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        print(f"DIGEST {name} M={M} var=0 {d0} rel={rel:.5f}" + (" WRONG" if rel > 0.02 else ""), flush=True)
        del ref
        ok = {}
        for v in VARS:
            dv = digest(v); ok[v] = dv == d0
            print(f"DIGEST {name} M={M} var={v} {dv} {'SAME' if ok[v] else 'WRONG'}", flush=True)
        if CHECKONLY: continue
        fa = mk(0)
        for v in VARS:
            if not ok[v]: continue
            fb = mk(v)
            for p in range(PAIRS):
                time.sleep(0.5)
                setv(0); a1 = run(fa, copies, SECS)
                setv(v); b1 = run(fb, copies, SECS)
                b2 = run(fb, copies, SECS)
                setv(0); a2 = run(fa, copies, SECS)
                A = (a1[0] + a2[0]) / 2; Bm = (b1[0] + b2[0]) / 2
                tf = lambda us: 2 * M * N * K / us / 1e6
                print(f"RES {name} {M} {v} {p} {a1[0]:.1f} {b1[0]:.1f} {b2[0]:.1f} {a2[0]:.1f} {Bm / A:.4f} "
                      f"{tf(A):.1f} {tf(Bm):.1f} {(a1[1] + a2[1]) / 2:.0f} {(b1[1] + b2[1]) / 2:.0f} "
                      f"{(a1[2] + a2[2]) / 2:.1f} {(b1[2] + b2[2]) / 2:.1f}", flush=True)
        setv(0)
    del real, Wv, Sv, RF, gu_w; torch.cuda.empty_cache()
stop[0] = True
print("DONE", flush=True)
