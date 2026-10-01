#!/usr/bin/env python3
"""Offline overlap/accuracy check for the int2 two-stage greedy lm_head.

Mirrors the methodology of sly/check_lmhead_int4.py, retargeted to what the two-stage
head actually needs to know, on real weights or synthetic logits:

  * coarse 2-bit logit error vs the fp32 exact reference,
  * coarse-top-K vs exact-top-K overlap (the candidate pool's quality),
  * exact-argmax-inside-coarse-top-K rate -- the correctness precondition: whenever it
    holds, the two-stage argmax IS the exact greedy argmax; the residual risk is the
    rows where it does not,
  * two-stage argmax (rerank over the coarse top-K) vs exact fp32 argmax agreement,
  * returned-mixture argmax vs the two-stage decision (interface self-check: the clamped
    row must encode the decision, never change it),
  * the production class path (process_weights_after_loading + apply) once per variant,
  * the sampling guard: with note_sampling() set, apply() must return the exact full
    row (valid to sample from) -- checked against the reference on every row.

Two modes:

  real      GPU + checkpoint snapshot (like check_lmhead_int4.py):
                PYTHONPATH=/opt/patches/sly/mxfp4 python3 sly/check_lmhead_int2.py \\
                    --snapshot /path/to/hf/snapshot [--rows 2048]
  synthetic  torch only, no checkpoint, CPU or CUDA:
                python3 sly/check_lmhead_int2.py --synthetic [--vocab 151936 --rows 256]

Falls back to synthetic automatically when no snapshot is found or no CUDA device is
present, so the numerics run anywhere; the timing section needs CUDA.

Exit code 0 iff every swept variant passes the gates: exact-top-1 coverage >=
--min-coverage (default 0.995) and coverage minus agreement <= 0.005 (the two-stage
decision may only disagree with the exact argmax where the argmax was not covered).
"""

import argparse
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mxfp4"))
import radiance_lmhead_int2 as I2  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(
        description="Offline overlap/accuracy check for the int2 two-stage greedy lm_head")
    ap.add_argument("--snapshot", default=None,
                    help="HF snapshot dir with *.safetensors holding lm_head.weight")
    ap.add_argument("--synthetic", action="store_true",
                    help="torch-only run: no checkpoint, no vllm, CPU or CUDA")
    ap.add_argument("--vocab", type=int, default=151936, help="synthetic vocab (rows of W)")
    ap.add_argument("--hidden", type=int, default=5120, help="synthetic hidden dim")
    ap.add_argument("--rows", type=int, default=None,
                    help="proxy hidden rows (default 2048 real / 256 synthetic)")
    ap.add_argument("--groups", default="128,64",
                    help="comma-separated group sizes to sweep")
    ap.add_argument("--topks", default="16", help="comma-separated candidate counts")
    ap.add_argument("--min-coverage", type=float, default=0.995,
                    help="gate: exact-top-1 inside coarse-top-K, per row")
    ap.add_argument("--no-timing", action="store_true", help="skip the CUDA timing sweep")
    return ap.parse_args()


def find_snapshot():
    for pat in ("~/.cache/huggingface/hub/models--*/snapshots/*/*.safetensors",
                "/root/.cache/huggingface/hub/models--*/snapshots/*/*.safetensors"):
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        if hits:
            return os.path.dirname(hits[0])
    return None


def load_real_weight(snapshot, device):
    from safetensors import safe_open

    for path in sorted(glob.glob(os.path.join(snapshot, "**", "*.safetensors"),
                                 recursive=True)):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith("lm_head.weight"):
                    w = f.get_tensor(key)
                    if not w.dtype.is_floating_point:
                        sys.exit(f"[check] {key} is {w.dtype}; need a floating-point lm_head")
                    print(f"[check] real lm_head: {key} {tuple(w.shape)} {w.dtype} "
                          f"from {os.path.basename(path)}")
                    return w.to(device)
    return None


def quantize_full(w, gs, chunk=8192):
    """Whole-matrix quantisation in row chunks (same transients as the class path)."""
    n, k = w.shape
    packed = torch.empty((n, k // 4), dtype=torch.uint8, device=w.device)
    scales = torch.empty((n, k // gs), dtype=torch.bfloat16, device=w.device)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        packed[i:j], scales[i:j] = I2.quantize_int2_rows(w[i:j], gs)
    return packed, scales


class FakeLayer(torch.nn.Module):
    """Just enough ParallelLMHead surface for the quant-method class path."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


def exact_reference(h2d, w, chunk_rows=8192):
    """fp32 exact logits [M, N], chunked over N so CPU never holds the fp32 weight."""
    m, k = h2d.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=h2d.device)
    hf = h2d.float()
    for i in range(0, n, chunk_rows):
        j = min(i + chunk_rows, n)
        out[:, i:j] = hf @ w[i:j].float().t()
    return out


def one_variant(w, packed, scales, gs, topk, proxies, min_coverage):
    """Full metric set for one (group size, topk) over every proxy. Returns ok."""
    n = packed.shape[0]
    r = min(topk, n)
    ok = True
    for name, h2d in proxies.items():
        ref = exact_reference(h2d, w)
        coarse = I2.coarse_logits(h2d, packed, scales, gs)
        exact_am = ref.argmax(dim=1)
        co_idx = coarse.topk(r, dim=1).indices
        ex_idx = ref.topk(r, dim=1).indices
        # set overlap per row: |exact-top-K ∩ coarse-top-K| / K
        co_mask = torch.zeros((h2d.shape[0], n), dtype=torch.bool, device=h2d.device)
        co_mask.scatter_(1, co_idx, True)
        ex_mask = torch.zeros((h2d.shape[0], n), dtype=torch.bool, device=h2d.device)
        ex_mask.scatter_(1, ex_idx, True)
        overlap = (co_mask & ex_mask).sum(dim=1).float().mean().item() / r
        covered = co_mask.gather(1, exact_am.unsqueeze(1)).squeeze(1).float().mean().item()
        # the two-stage decision: rerank the coarse candidates exactly
        mixture, cand_idx, cand = I2.greedy_two_stage(h2d, packed, scales, gs, topk,
                                                      (w, None))
        # two-stage decision: argmax over the exactly-scored candidates, mapped back
        # to token ids (the pool is sorted by coarse rank, positions are not ids)
        two_am = cand_idx.gather(1, cand.argmax(dim=1, keepdim=True)).squeeze(1)
        mix_am = mixture.argmax(dim=1)
        agree = (two_am == exact_am).float().mean().item()
        iface = (mix_am == two_am).float().mean().item()
        err = coarse - ref
        rms = err.pow(2).mean().sqrt().item()
        rel = rms / ref.pow(2).mean().sqrt().item()
        m = h2d.shape[0]
        cov_gap = covered - agree
        gate_ok = covered >= min_coverage and cov_gap <= 0.005 and iface >= 0.999
        ok &= gate_ok
        print(f"    {name:<6} logit_err_rms={rms:8.4f} ({rel * 100:5.2f}% of ref rms)  "
              f"top{r}_overlap={overlap * 100:6.2f}%  covered={covered * 100:6.2f}%  "
              f"agree={agree * 100:6.2f}% ({int(m - int(round(agree * m)))}/{m} flips)  "
              f"mixture={iface * 100:6.2f}%  {'OK' if gate_ok else 'FAIL'}")
        # interface contract: every non-candidate entry is clamped below the rerank
        # minimum, i.e. the mixture can never promote a row the rerank did not score
        floor = cand.min(dim=1, keepdim=True).values
        noncand = mixture.masked_fill(co_mask, float("-inf"))
        assert bool((noncand <= floor).all()), "mixture leaked above the rerank minimum"
        del noncand
        del ref, coarse, co_mask, ex_mask, mixture, cand, cand_idx
    return ok


def timing_sweep(w, packed, scales, gs, topk, device):
    if not device.startswith("cuda"):
        return
    n, k = w.shape
    bytes_coarse = packed.numel() + scales.numel() * 2
    print(f"    [timing] stage-1 weight traffic {bytes_coarse / 1e6:.1f} MB/call "
          f"(vs {n * k * 2 / 1e6:.0f} MB bf16)")
    for m in (1, 8, 64, 256):
        h = torch.randn(m, k, dtype=torch.bfloat16, device=device)

        def stage1():
            return I2.coarse_logits(h, packed, scales, gs)

        def both():
            mixture, _, _ = I2.greedy_two_stage(h, packed, scales, gs, topk, (w, None))
            return mixture

        def stock():  # what the engine's bf16 head does today
            return torch.matmul(h, w.t())

        for fn in (stage1, both, stock):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 20 * 1e3
            print(f"    M={m:<4} {fn.__name__:<7} {dt:8.3f} ms")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.no_timing:
        args.no_timing = True

    snapshot = args.snapshot
    if not args.synthetic and snapshot is None:
        snapshot = find_snapshot()
    w = None
    if not args.synthetic and snapshot:
        w = load_real_weight(snapshot, device)
        if w is None:
            print("[check] no lm_head.weight in snapshot; falling back to synthetic")
    if w is None:
        if not args.synthetic:
            print("[check] no snapshot found; running synthetic (pass --snapshot for "
                  "real weights)")
        vocab, hidden = args.vocab, args.hidden
        torch.manual_seed(20260901)
        w = (torch.randn(vocab, hidden) * 0.02).to(torch.bfloat16).to(device)
        print(f"[check] synthetic lm_head: [{vocab}, {hidden}] bf16 on {device}")

    n, k = w.shape
    if k % 4 or any(int(g) <= 0 or k % int(g) for g in args.groups.split(",")):
        sys.exit(f"[check] hidden {k} incompatible with groups {args.groups}")
    rows = args.rows or (512 if n > 100000 else 256)  # real weights: keep transients ~GBs
    if device == "cpu" and rows > 512:
        rows = 256
        print(f"[check] CPU run: capping proxy rows to {rows}")
    # column-magnitude-aware hidden scale, accumulated in row chunks (no whole-weight
    # fp32 transient)
    abs_sum = torch.zeros(k, device=device)
    for i in range(0, n, 8192):
        j = min(i + 8192, n)
        abs_sum += w[i:j].float().abs().sum(dim=0)
    g_scale = 1.0 + abs_sum / n
    del abs_sum

    torch.manual_seed(20260902)
    proxies = {}
    proxies["rand"] = (torch.randn(rows, k) * g_scale.unsqueeze(0)).to(torch.bfloat16).to(device)
    # embedding-row proxy: late-layer hiddens cluster around embedding directions; these
    # make the argmax non-trivial (the exact winner is a near-neighbor of many rows)
    rows_of_w = w[torch.randint(0, n, (rows,), device=device)].float()
    rows_of_w = rows_of_w / rows_of_w.norm(dim=1, keepdim=True) * (k ** 0.5)
    proxies["embrow"] = (rows_of_w * g_scale.unsqueeze(0)).to(torch.bfloat16).to(device)

    all_ok = True
    for gs in (int(g) for g in args.groups.split(",")):
        packed, scales = quantize_full(w, gs)
        for topk in (int(t) for t in args.topks.split(",")):
            print(f"  [group {gs}, top-{topk}]")
            all_ok &= one_variant(w, packed, scales, gs, topk, proxies, args.min_coverage)
            if not args.no_timing:
                timing_sweep(w, packed, scales, gs, topk, device)

    # the production class path, once, at the default knobs: process + apply must agree
    # with the function path, keep the source weight, and honor the sampling guard.
    print("  [class path]")
    saved_gs, saved_topk = I2.GROUP_SIZE, I2.TOPK
    I2.GROUP_SIZE, I2.TOPK = int(args.groups.split(",")[0]), int(args.topks.split(",")[0])
    try:
        method = I2.RadianceLMHeadInt2()
        layer = FakeLayer(w.clone())
        method.process_weights_after_loading(layer)
        assert layer.weight.dtype == w.dtype and layer.weight.equal(w), \
            "the exact rerank source must stay loaded"
        assert layer.weight_int2.dtype == torch.uint8
        assert layer.weight_int2_scale.dtype == torch.bfloat16
        assert layer.weight_int2.shape == (n, k // 4)
        assert layer.weight_int2_scale.shape == (n, k // I2.GROUP_SIZE)
        h2d = proxies["embrow"][:64].reshape(8, 8, k)  # 3-D input: reshape path too
        ref_am = exact_reference(h2d.reshape(-1, k), w).argmax(dim=1)
        out = method.apply(layer, h2d)
        assert out.shape == (8, 8, n) and out.dtype == h2d.dtype
        fn_mixture, _, _ = I2.greedy_two_stage(h2d.reshape(-1, k), layer.weight_int2,
                                               layer.weight_int2_scale, I2.GROUP_SIZE,
                                               I2.TOPK, (layer.weight, None))
        assert out.reshape(-1, n).equal(fn_mixture.to(out.dtype)), \
            "apply() must encode exactly the two-stage mixture"
        agree = (out.reshape(-1, n).argmax(dim=1) == ref_am).float().mean().item()
        print(f"    apply() dtype/shape ok, mixture identical, argmax agree "
              f"{agree * 100:.2f}% vs exact on the embrow proxy")
        # sampling guard: the fallback must be exactly the full row, valid to sample
        I2.note_sampling()
        out = method.apply(layer, h2d)
        exact_out = torch.matmul(h2d.reshape(-1, k), layer.weight.t())
        assert out.reshape(-1, n).equal(exact_out), \
            "under sampling apply() must return the exact full-width row"
        I2.reset_sampling()
        print("    sampling guard ok: note_sampling() -> exact full row")
    finally:
        I2.GROUP_SIZE, I2.TOPK = saved_gs, saved_topk
        I2.reset_sampling()

    print(f"[check] {'GO' if all_ok else 'NO-GO'}: gates were exact-top1 coverage >= "
          f"{args.min_coverage * 100:.1f}% and coverage-agreement <= 0.5%")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
