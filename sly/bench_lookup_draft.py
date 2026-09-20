#!/usr/bin/env python3
"""Card check and timing of radiance_lookup_draft's kernels (throw-away container, GPU exclusive, no model).

  bench_lookup_draft.py --check    the scan / apply kernels against the pure-python reference: match, decision,
                                   draft tokens, the cached draft distribution, the per-slot state; slots that are
                                   not the batch order, a padded (-1) row, planted repeats, hot / not hot
  bench_lookup_draft.py --time     what one override costs (scan + apply, python launch included) at 1k .. 262k of
                                   history, 1 and 8 requests, history in UVA host memory (what vLLM's all_token_ids
                                   is) and on the device

  bench_lookup_draft.py --rejection  vLLM's V2 rejection sampler with the point-mass draft distribution the override
                                   writes: the sampled tokens must follow the target distribution (lossless), for a
                                   deterministic draft that is right and, as a control with power, one that is wrong

BENCH_DEVICE=cpu TRITON_INTERPRET=1 runs --check on the CPU through Triton's interpreter (no GPU; slow; device
history only): catches compile / semantics errors before a card window.
"""
import argparse
import os
import random
import sys
import time
import types
from pathlib import Path

os.environ["RADIANCE_LOOKUP_DRAFT"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import radiance_lookup_draft as R  # noqa: E402

DEVICE = os.environ.get("BENCH_DEVICE", "cuda")
K = 7
TOP_K = 4
VOCAB = 4096
MAX_REQS = 8


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


class Rig:
    """The pieces of vLLM's runner the override touches, with real tensors."""

    def __init__(self, uva=True, max_len=262144, probabilistic=True, min_dist=0):
        self.dev = torch.device(DEVICE)
        self.max_len = max_len
        if uva and DEVICE == "cuda":
            from vllm.v1.worker.gpu.buffer_utils import UvaBuffer
            buf = UvaBuffer((MAX_REQS, max_len), torch.int32)
            self.ids_cpu, self.ids = buf.cpu, buf.uva
        else:
            self.ids = torch.zeros(MAX_REQS, max_len, dtype=torch.int32, device=self.dev)
            self.ids_cpu = None
        self.lens = torch.zeros(MAX_REQS, dtype=torch.int32, device=self.dev)
        self.rs = types.SimpleNamespace(all_token_ids=types.SimpleNamespace(gpu=self.ids),
                                        total_len=types.SimpleNamespace(gpu=self.lens))
        self.sp = types.SimpleNamespace(
            draft_tokens=torch.zeros(MAX_REQS, K, dtype=torch.int64, device=self.dev),
            draft_logits=(torch.full((MAX_REQS, K, VOCAB), -float("inf"), dtype=torch.float32, device=self.dev)
                          if probabilistic else None),
            _cached_candidate_ids=torch.zeros(MAX_REQS, K, TOP_K, dtype=torch.int64, device=self.dev),
            selector_top_k=TOP_K, num_speculative_steps=K,
            vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(get_vocab_size=lambda: VOCAB)))
        R._state.update(ready=False, failed=False, logged=True, calls=0)
        R._prepare(self.sp, self.rs)
        R._state["min_dist"] = min_dist
        self.min_dist = min_dist

    def set_history(self, slot, tokens):
        t = torch.as_tensor(tokens, dtype=torch.int32)
        if self.ids_cpu is not None:
            self.ids_cpu[slot, :len(t)] = t
        else:
            self.ids[slot, :len(t)] = t.to(self.dev)
        self.lens[slot] = len(t)

    def fake_dflash_draft(self, rng):
        """What the graph leaves behind: random tokens and a candidate-shaped cached distribution."""
        sp = self.sp
        sp.draft_tokens.copy_(torch.randint(0, VOCAB, (MAX_REQS, K), dtype=torch.int64, generator=rng, device="cpu"))
        if sp.draft_logits is not None:
            sp.draft_logits.fill_(-float("inf"))
            cand = torch.randint(0, VOCAB, (MAX_REQS, K, TOP_K), dtype=torch.int64, generator=rng, device="cpu")
            sp._cached_candidate_ids.copy_(cand)
            score = torch.randn(MAX_REQS, K, TOP_K, generator=rng)
            for s in range(MAX_REQS):
                for k in range(K):
                    sp.draft_logits[s, k, cand[s, k].to(self.dev)] = score[s, k].to(self.dev)

    def call(self, slots, num_sampled, bound):
        n = len(slots)
        ib = types.SimpleNamespace(num_reqs=n, idx_mapping=torch.tensor(slots, dtype=torch.int32, device=self.dev),
                                   seq_lens_cpu_upper_bound=torch.full((n,), bound, dtype=torch.int32))
        ns = torch.tensor(num_sampled, dtype=torch.int32, device=self.dev)
        R._override(self.sp, self.rs, ib, ns)
        sync()


def planted_history(rng, length, source_len=400):
    """Random tokens with a block copied from far back, ending so that the suffix continues the copy."""
    h = [rng.randrange(VOCAB) for _ in range(length)]
    if length > 3 * source_len:
        src = rng.randrange(0, length // 2)
        cut = rng.randrange(20, source_len)
        tail = h[src:src + cut]
        h[length - len(tail):] = tail
    return h


def extend(rng, h, plan, p1, clen1):
    """The tokens a request appends in one step. `plan`: 'copy' (the lookup's continuation was right, then the
    bonus token continues it), 'short' (the step ends in a 4-token repeat of something far back), 'random'."""
    a = rng.choice([0, 2, 3, 6])
    if plan == "copy" and clen1 > 0:
        a = min(a, clen1 - 1)
        return h + h[p1:p1 + a + 1], a + 1
    if plan == "short" and len(h) > 100:
        a = rng.choice([3, 6])
        q = rng.randrange(0, len(h) - 10)
        return h + [rng.randrange(VOCAB) for _ in range(a + 1 - 4)] + h[q:q + 4], a + 1
    return h + [rng.randrange(VOCAB) for _ in range(a + 1)], a + 1


def check():
    rng = random.Random(7)
    grng = torch.Generator().manual_seed(11)
    bad = total = 0
    hot_seen = used_seen = 0
    rigs = [(uva, probabilistic, md) for uva in ((True, False) if DEVICE == "cuda" else (False,))
            for probabilistic in (True, False) for md in (0, 1500)]
    for uva, probabilistic, md in rigs:
        if True:
            rig = Rig(uva=uva, max_len=65536, probabilistic=probabilistic, min_dist=md)
            state = {}                                             # python mirror of st_len / st_used per slot
            for rnd in range(5):
                for slots in ([0], [3, 1, 6], [5, 2, -1], [7, 4, 0, 2, 1, 6, 3, 5]):
                    n = len(slots)
                    hist, plan1 = {}, {}
                    for s in slots:
                        if s >= 0:
                            hist[s] = planted_history(rng, rng.choice([700, 2500, 9000, 30000]))
                            rig.set_history(s, hist[s])
                            state[s] = (-1, 0)
                            R._state["st_len"][s] = -1
                            R._state["st_used"][s] = 0
                    p1 = {}
                    for step in (1, 2):
                        ns = [1] * n
                        if step == 2:
                            for r, s in enumerate(slots):
                                if s >= 0:
                                    plan = rng.choice(["copy", "copy", "short", "random"])
                                    hist[s], ns[r] = extend(rng, hist[s], plan, *p1[s])
                                    rig.set_history(s, hist[s])
                        rig.fake_dflash_draft(grng)
                        before_tokens = rig.sp.draft_tokens.clone().cpu()
                        before_logits = rig.sp.draft_logits.clone().cpu() if probabilistic else None
                        before_cached = rig.sp._cached_candidate_ids.clone().cpu()
                        rig.call(slots, ns, max(len(h) for h in hist.values()) + 1)
                        tokens = rig.sp.draft_tokens.cpu()
                        logits = rig.sp.draft_logits.cpu() if probabilistic else None
                        cached = rig.sp._cached_candidate_ids.cpu()
                        st_len, st_used = R._state["st_len"].cpu(), R._state["st_used"].cpu()
                        leftover = int(R._state["best"].abs().sum())
                        for r, s in enumerate(slots):
                            total += 1
                            if s < 0:
                                ok = torch.equal(tokens[r], before_tokens[r])
                            else:
                                h = hist[s]
                                L = len(h)
                                m, p = R.ref_best_match(h, L, min(R.ENTER, R.STAY), min_dist=md)
                                prev_len, prev_used = state[s]
                                use, clen, hot = R.ref_decide(m, p, L, K, prev_used, prev_len, ns[r])
                                hot_seen += hot and use
                                used_seen += use
                                p1[s] = (p, clen)
                                exp_tokens = before_tokens[r].clone()
                                for k in range(clen):
                                    exp_tokens[k] = h[p + k]
                                ok = torch.equal(tokens[r], exp_tokens)
                                if probabilistic:
                                    for k in range(K):
                                        if k < clen:
                                            row = logits[s, k]
                                            fin = torch.isfinite(row).nonzero().flatten().tolist()
                                            ok &= fin == [h[p + k]] and float(row[h[p + k]]) == 0.0
                                            ok &= bool((cached[s, k] == h[p + k]).all())
                                        else:
                                            ok &= torch.equal(logits[s, k], before_logits[s, k])
                                            ok &= torch.equal(cached[s, k], before_cached[s, k])
                                ok &= int(st_len[s]) == L and int(st_used[s]) == int(use)
                                state[s] = (L, int(use))
                            ok &= leftover == 0
                            if not ok:
                                bad += 1
                                print(f"MISMATCH uva={uva} prob={probabilistic} min_dist={md} round {rnd} step {step} slots {slots} row {r}")
    print(f"{total} request-steps checked (UVA and device history, probabilistic and greedy draft, min_dist 0 and 1500; lookup taken in "
          f"{used_seen}, of those {hot_seen} through the hot / stay rule), {bad} mismatches")
    return bad == 0 and used_seen > 0 and hot_seen > 0


def check_rejection(n=200000, vocab=16, k=2):
    """Speculative sampling with a deterministic (point-mass) draft: the first token's distribution is the target's
    p0, given an accepted first token the second's is p1, given both the bonus token's is p2. Drafts and target rows
    are the same for every request; the seeds differ."""
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    dev = torch.device(DEVICE)
    g = torch.Generator().manual_seed(5)
    base = torch.randn(k + 1, vocab, generator=g) * 1.2
    probs = torch.softmax(base, -1)
    drafts = [int(probs[0].argsort()[-2]), int(probs[1].argsort()[-2])]     # second most likely at both steps: accepted about half the time
    tokens = torch.tensor([1] + drafts).repeat(n).to(dev)                   # [last sampled, d1, d2] per request
    target = base.repeat(n, 1).to(dev)
    cu = (torch.arange(n + 1, dtype=torch.int32) * (k + 1)).to(dev)
    pos = (torch.arange(k + 1, dtype=torch.int64) + 100).repeat(n).to(dev)
    idx = torch.arange(n, dtype=torch.int32).to(dev)
    eidx = idx.repeat_interleave(k + 1)
    elp = torch.arange(k + 1, dtype=torch.int32).repeat(n).to(dev)
    temperature = torch.ones(n, dtype=torch.float32, device=dev)
    seed = torch.randint(1, 2 ** 62, (n,), generator=g, dtype=torch.int64).to(dev)

    def run(draft_logits):
        out, num = rejection_sample(target, draft_logits, tokens, cu, pos, idx, eidx, elp, temperature, seed, k)
        return out.cpu(), num.cpu()

    def point_mass(at):
        dl = torch.full((n, k, vocab), -float("inf"), dtype=torch.float32, device=dev)
        for step, t in enumerate(at):
            dl[:, step, t] = 0.0
        return dl

    def zmax(counts, total, pvec):
        exp = pvec * total
        return float(((counts - exp) / (exp * (1 - pvec)).clamp(min=1e-9).sqrt()).abs().max())

    ok = True
    print(f"target p0 at draft 1 = {probs[0][drafts[0]]:.3f}, p1 at draft 2 = {probs[1][drafts[1]]:.3f}")
    for name, dl, want in (("no draft distribution (vLLM's deterministic path)", None, True),
                           ("point mass on the draft tokens (what the override writes)", point_mass(drafts), True),
                           ("control: point mass on the WRONG tokens", point_mass([(d + 1) % vocab for d in drafts]), False)):
        out, num = run(dl)
        z = []
        c0 = torch.bincount(out[:, 0], minlength=vocab).float()
        z.append(zmax(c0, n, probs[0]))
        acc1 = out[:, 0] == drafts[0]
        m1 = acc1 & (num >= 2)
        c1 = torch.bincount(out[m1, 1], minlength=vocab).float()
        z.append(zmax(c1, int(m1.sum()), probs[1]))
        m2 = m1 & (out[:, 1] == drafts[1]) & (num >= 3)
        c2 = torch.bincount(out[m2, 2], minlength=vocab).float()
        z.append(zmax(c2, int(m2.sum()), probs[2]))
        passed = max(z) < 4.5
        verdict = "OK" if passed == want else "UNEXPECTED"
        ok &= passed == want
        print(f"  {name}: max |z| first / second / third token = {z[0]:.1f} / {z[1]:.1f} / {z[2]:.1f} "
              f"(accept 1st {float(acc1.float().mean()):.3f}) -> {'lossless' if passed else 'biased'}  [{verdict}]")
    return ok


def timing(reps=200):
    print(f"{'history':>8} {'reqs':>4} {'memory':>6} | {'gpu us':>8} {'wall us':>8}")
    for ctx in (1024, 8192, 33000, 99000, 262000):
        for uva in (True, False):
            rig = Rig(uva=uva)
            rng = random.Random(3)
            for n in (1, 8):
                slots = list(range(n))
                for s in slots:
                    rig.set_history(s, [rng.randrange(VOCAB) for _ in range(ctx)])
                rig.fake_dflash_draft(torch.Generator().manual_seed(1))
                for _ in range(20):
                    rig.call(slots, [4] * n, ctx + 2)
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ib = types.SimpleNamespace(num_reqs=n, idx_mapping=torch.tensor(slots, dtype=torch.int32, device=rig.dev),
                                           seq_lens_cpu_upper_bound=torch.full((n,), ctx + 2, dtype=torch.int32))
                ns = torch.tensor([4] * n, dtype=torch.int32, device=rig.dev)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ev0.record()
                for _ in range(reps):
                    R._override(rig.sp, rig.rs, ib, ns)
                ev1.record()
                wall = (time.perf_counter() - t0) / reps * 1e6      # launches only, not waiting for the GPU
                torch.cuda.synchronize()
                gpu = ev0.elapsed_time(ev1) / reps * 1e3
                print(f"{ctx:>8} {n:>4} {'uva' if uva else 'device':>6} | {gpu:8.1f} {wall:8.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--rejection", action="store_true")
    a = ap.parse_args()
    ok = True
    if a.check:
        ok = check()
    if a.rejection:
        ok = check_rejection() and ok
    if a.time:
        timing()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
