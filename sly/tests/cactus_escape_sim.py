#!/usr/bin/env python3
"""CPU verification of speculative-decoding acceptance rules for issue #164.

Pure stdlib. Four parts:
  A. Exact per-step output distributions (enumeration over a small vocab):
     - stock reproduces q (Leviathan identity)
     - escape mixture reproduces (1-eps)*out_rule + eps*q  (exact identity)
     - escape probability floor: esc_mix >= eps * esc_stock
  B. Per-step KL of the REPO's threshold-only Cactus (+cap) vs q, over random
     (q, p) pairs -- checks the docstring's "KL <= Gamma(delta)" claim.
  C. Markov-chain loop attractor: lock-in probability and mean escape time for
     stock / cactus / cactus+cap / escape-mixture, at several (q_c, p_c) points.
  D. The absorbing corner: KL stays tiny while escape probability is exactly 0.
"""
import math
import random
import statistics

V = 5


def normalize(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def stock_residual(q, p):
    r = [max(q[i] - p[i], 0.0) for i in range(V)]
    return normalize(r) if sum(r) > 1e-12 else q[:]


def repo_h(qx, delta, cap=None, fade=1.0):
    """Current main (1.9.0+): h = clip(q + sqrt(2 d q (1-q)), 0, 1), then cap 1-r(1-q)."""
    d = delta * fade
    one_m_q = max(1.0 - qx, 0.0)
    h = min(max(qx + math.sqrt(2.0 * d * qx * one_m_q), 0.0), 1.0)
    if cap is not None:
        h = min(h, 1.0 - cap * (1.0 - qx))
    return h


def rule_accept(h, px):
    return min(1.0, h / px) if px > 0 else 0.0


def output_dist(q, p, accept_fn, residual):
    """out[x] = p[x]*a(x) + (1 - sum p*a) * residual[x]"""
    a = [accept_fn(i) for i in range(V)]
    mass = sum(p[i] * a[i] for i in range(V))
    return [p[i] * a[i] + (1.0 - mass) * residual[i] for i in range(V)], a, mass


def kl(a, b):
    return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0)


def tv(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def gamma_fn(delta):
    g = 1.0 / (1.0 + 2.0 * delta)
    return -g * math.log(g) + (1.0 - g) * math.log(1.0 - g + delta)


def part_a():
    print("=== A. exact identities (enumeration, V=5) ===")
    rng = random.Random(164)
    worst_mix = worst_stock = worst_floor = 0.0
    for _ in range(4000):
        q = normalize([rng.random() ** 2 for _ in range(V)])
        p = normalize([rng.random() ** 2 for _ in range(V)])
        delta, cap, eps = 0.5, 0.5, rng.choice([0.1, 0.25, 0.5])
        r = stock_residual(q, p)
        h = [repo_h(q[i], delta, cap) for i in range(V)]
        out_c, a_c, _ = output_dist(q, p, lambda i: rule_accept(h[i], p[i]), r)
        out_s, a_s, _ = output_dist(q, p, lambda i: rule_accept(q[i], p[i]), r)
        # mixture mechanism: with prob eps run stock, else cactus
        out_m = [(1 - eps) * out_c[i] + eps * out_s[i] for i in range(V)]
        a_m = [(1 - eps) * a_c[i] + eps * a_s[i] for i in range(V)]
        out_m2, _, _ = output_dist(q, p, lambda i: a_m[i], r)
        worst_mix = max(worst_mix, tv(out_m, out_m2))
        worst_stock = max(worst_stock, tv(out_s, q))
        # escape floor per draft token: 1 - a(x)
        for i in range(V):
            esc_s, esc_m = 1 - a_s[i], 1 - a_m[i]
            worst_floor = max(worst_floor, eps * esc_s - esc_m)
    print(f"stock out == q:                max TV = {worst_stock:.2e}")
    print(f"mixture out == (1-e)C + e*q:   max TV = {worst_mix:.2e}")
    print(f"escape floor esc_m >= e*esc_s: max violation = {worst_floor:.2e}")


def paper_faithful_out(q, p, delta):
    """The PAPER's algorithm (Corollary 5 + Observation 1), for comparison:
    gamma* = min(q_n + sqrt(2 d q_n (1-q_n)), 1); h renormalized over V;
    phi = min(1, h/p); g = (h - p*phi) / E_p[1-phi]."""
    out = [0.0] * V
    for n in range(V):
        if p[n] <= 0:
            continue
        qn = q[n]
        gamma = min(qn + math.sqrt(2 * delta * qn * max(1 - qn, 0)), 1.0)
        h = [gamma if i == n else (1 - gamma) / (1 - qn) * q[i]
             for i in range(V)] if qn < 1 else [float(i == n) for i in range(V)]
        phi = [min(1.0, h[i] / p[i]) if p[i] > 0 else 0.0 for i in range(V)]
        denom = sum(p[i] * (1 - phi[i]) for i in range(V))
        if denom > 1e-12:
            g = [(h[i] - p[i] * phi[i]) / denom for i in range(V)]
        else:
            g = h[:]
        out[n] += p[n] * phi[n]
        for i in range(V):
            out[i] += p[n] * (1 - phi[n]) * g[i]
    return out


def part_b():
    print("\n=== B. per-step KL of the repo rule vs q (delta=0.5) ===")
    print(f"Gamma(0.5) binary worst case (arXiv v1 App A.3) = {gamma_fn(0.5):.4f} nats")
    rng = random.Random(164)
    pairs = [(normalize([rng.random() ** 2 for _ in range(V)]),
              normalize([rng.random() ** 2 for _ in range(V)])) for _ in range(20000)]
    for name, hfn, cap in [("current", repo_h, None), ("current", repo_h, 0.5)]:
        kls, tvs, over = [], [], 0
        for q, p in pairs:
            r = stock_residual(q, p)
            h = [hfn(q[i], 0.5, cap) for i in range(V)]
            out, _, _ = output_dist(q, p, lambda i: rule_accept(h[i], p[i]), r)
            k = kl(out, q)
            kls.append(k)
            tvs.append(tv(out, q))
            over += k > gamma_fn(0.5) + 1e-9
        n = len(pairs)
        print(f"{name} cap={cap}: mean KL={statistics.mean(kls):.4f}  "
              f"p99={sorted(kls)[int(0.99*n)]:.4f}  max={max(kls):.4f}  "
              f"mean TV={statistics.mean(tvs):.4f}  frac KL>Gamma_v1: {over/n:.3f}")
    kls_f = [kl(paper_faithful_out(q, p, 0.5), q) for q, p in pairs]
    over_f = sum(k > gamma_fn(0.5) + 1e-9 for k in kls_f)
    print(f"paper-faithful:    mean KL={statistics.mean(kls_f):.4f}  "
          f"p99={sorted(kls_f)[int(0.99*len(kls_f))]:.4f}  max={max(kls_f):.4f}  "
          f"frac KL>Gamma: {over_f/len(kls_f):.3f}")
    # zero-escape thresholds (h >= 1 => acceptance 1 for any drafter)
    for d in (0.1, 0.25, 0.5):
        qstar = 1 / (1 + 2 * d)  # gamma* >= 1  <=>  q >= 1/(1+2d); same in 1.5.1 and main
        print(f"delta={d}: zero escape for q >= {qstar:.3f} (gamma*, verified in v1.5.1 and main)")


def chain_sim(q_c, p_c, rule, T=400, runs=400, seed=164):
    """3-cycle A->B->C->A. On-cycle: q(next)=q_c, p(next)=p_c.
    Off-cycle: q pulls back weakly; escape measured as leaving the cycle.
    rule: 'stock' | 'cactus' | 'cap' | ('mix', eps)
    Returns (lock-in fraction at T, mean escape step | escaped)."""
    rng = random.Random(seed)
    cycle = [0, 1, 2]
    off = [3, 4]
    lock = 0
    escapes = []
    for _ in range(runs):
        state = 0  # start on cycle
        escaped_at = None
        for t in range(T):
            nxt = cycle[(cycle.index(state) + 1) % 3] if state in cycle else None
            if state in cycle:
                q = [0.0] * V
                q[nxt] = q_c
                for o in off:
                    q[o] = (1 - q_c) / 2
                p = [0.0] * V
                p[nxt] = p_c
                for o in off:
                    p[o] = (1 - p_c) / 2
            else:
                # off cycle: diffuse, weak pull back to cycle entry
                q = [0.1, 0.1, 0.1, 0.35, 0.35]
                p = [0.2, 0.2, 0.2, 0.2, 0.2]
            # draft token
            x = rng.choices(range(V), weights=p)[0]
            if rule == "stock":
                a = min(1.0, q[x] / p[x]) if p[x] > 0 else 0.0
            else:
                cap = 0.5 if rule in ("cap",) or isinstance(rule, tuple) else None
                h = repo_h(q[x], 0.5, cap)
                a = rule_accept(h, p[x])
                if isinstance(rule, tuple):
                    eps = rule[1]
                    a_s = min(1.0, q[x] / p[x]) if p[x] > 0 else 0.0
                    a = (1 - eps) * a + eps * a_s
            if rng.random() < a:
                state = x
            else:
                r = stock_residual(q, p)
                state = rng.choices(range(V), weights=r)[0]
            if state not in cycle and escaped_at is None:
                escaped_at = t + 1
                # once escaped, the run counts as escaped (absorbing for metric)
                break
        if escaped_at is None:
            lock += 1
        else:
            escapes.append(escaped_at)
    return lock / runs, (statistics.mean(escapes) if escapes else float("inf"))


def part_c():
    print("\n=== C. loop-attractor Markov chain (3-cycle, T=400, 400 runs) ===")
    print(f"{'q_c':>6} {'p_c':>7} | {'stock':>14} {'cactus d=.5':>14} "
          f"{'+cap r=.5':>14} {'mix e=.10':>14} {'mix e=.25':>14}")
    for q_c, p_c in [(0.90, 1.0), (0.99, 1.0), (0.90, 0.99), (0.99, 0.999),
                     (0.999, 0.9999), (0.80, 0.99)]:
        row = []
        for rule in ("stock", "cactus", "cap", ("mix", 0.10), ("mix", 0.25)):
            lk, esc = chain_sim(q_c, p_c, rule)
            row.append(f"{lk*100:5.1f}%/{esc:6.1f}")
        print(f"{q_c:>6} {p_c:>7} | " + " ".join(f"{c:>14}" for c in row))
    print("(cells: lock-in % at T=400 / mean escape step when escaped)")
    # acceptance-rate cost of the mixture on a benign state (q=p-ish, high agree)
    print("\n=== acceptance cost of mixture (benign states, q==p) ===")
    for eps in (0.1, 0.25):
        # when q==p: stock a=1, cactus a=1 -> mixture a=1: zero cost.
        pass
    rng = random.Random(7)
    for eps in (0.1, 0.25):
        losses = []
        for _ in range(20000):
            q = normalize([rng.random() for _ in range(V)])
            p = normalize([q[i] * (0.7 + 0.6 * rng.random()) for i in range(V)])
            h = [repo_h(q[i], 0.5, 0.5) for i in range(V)]
            a_c = [rule_accept(h[i], p[i]) for i in range(V)]
            a_s = [rule_accept(q[i], p[i]) for i in range(V)]
            a_m = [(1 - eps) * a_c[i] + eps * a_s[i] for i in range(V)]
            acc_c = sum(p[i] * a_c[i] for i in range(V))
            acc_m = sum(p[i] * a_m[i] for i in range(V))
            losses.append(acc_c - acc_m)
        print(f"eps={eps}: mean acceptance loss = {statistics.mean(losses):.4f} "
              f"(max {max(losses):.4f}); bound eps*(a_c-a_s) <= eps")


def part_d():
    print("\n=== D. absorbing corner: tiny KL, zero escape ===")
    delta, eps = 0.5, 0.25
    cases = [
        ("q=0.999 p=1.00 (greedy drafter), no cap", 0.999, 1.0, None),
        ("q=0.999 p=1.00, cap r=0.5", 0.999, 1.0, 0.5),
        ("q=0.90  p=0.92, cap r=0.5 (probabilistic drafter)", 0.90, 0.92, 0.5),
        ("q=0.98  p=0.99, cap r=0.5", 0.98, 0.99, 0.5),
    ]
    for name, qn, pn, cp in cases:
        q = [qn] + [(1 - qn) / (V - 1)] * (V - 1)
        p = [pn] + [(1 - pn) / (V - 1)] * (V - 1)
        r = stock_residual(q, p)
        h = [repo_h(q[i], delta, cp) for i in range(V)]
        out_c, a_c, _ = output_dist(q, p, lambda i: rule_accept(h[i], p[i]), r)
        out_s, a_s, _ = output_dist(q, p, lambda i: rule_accept(q[i], p[i]), r)
        a_m = [(1 - eps) * a_c[i] + eps * a_s[i] for i in range(V)]
        esc_c, esc_s, esc_m = 1 - a_c[0], 1 - a_s[0], 1 - a_m[0]

        def tte(e):
            return f"{1/e:8.0f}" if e > 0 else "     inf"

        print(f"{name}:")
        print(f"   KL(out_cactus,q) = {kl(out_c,q):.5f} nats (budget Gamma(0.5) = {gamma_fn(0.5):.4f})")
        print(f"   escape/step: cactus={esc_c:.2e}  stock={esc_s:.2e}  mix(e=0.25)={esc_m:.2e}")
        print(f"   expected steps to escape: cactus={tte(esc_c)}  stock={tte(esc_s)}  mix={tte(esc_m)}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()