"""Adaptive verify width for the DFlash2 block drafter (RADIANCE_ADAPTIVE_WIDTH=0|uniform|perseq).

Port of Radiance core/sched/adaptive_k.{h,cpp} (commit 60894df). The block drafter emits all
`depth` tokens in one pass whatever is verified, so the verify width only decides how many target
rows a step spends (and how many context rows the drafter's K/V pass takes, which mirrors the
target step). Fewer rows on positions that were going to be rejected shorten the step; every
emitted token is still the target's own choice (lossless).

THE MODEL. A request accepts each further draft position with probability q (geometric). Verifying
k drafts yields E(k) = 1 + sum_{j=1..k} q^j tokens for 1 + k rows. The step costs
T(M) = a + b*M ms with M = total verify rows. The batch maximises sum_i E_i(k_i) / T(M).
 - perseq : k_i per request from a small set (default 1,3,5,7), greedy marginal gain per extra row,
            total rows restricted to multiples of QUANT (the cudagraph capture sizes), stop where
            tokens/T(M) peaks.
 - uniform: ONE width for the whole batch from the same set, argmax of the same objective. The
            batch stays a uniform decode batch, so vLLM keeps the FULL cudagraph (see
            patch_adaptive_width.py: one extra graph set per width).
Nothing is shrunk while the unshrunk row total is below MIN_ROWS (weight-bound: a draft row is
nearly free there, and a single stream stays bit-identical).

q is estimated per request with censoring: a verify of k drafts that took `a` of them evaluated
a + (a < k) positions, q = hits / evaluated over exponentially decayed sums (ALPHA), shrunk
toward the pooled estimate with PRIOR pseudo-positions so a new request starts at the batch mean.

Pure host code, no torch import. Knobs (all optional):
  RADIANCE_ADAPTIVE_WIDTH   0 (default) | uniform | perseq
  RADIANCE_AW_LENS          "1,3,5,7"  widths tried (the full depth is always added)
  RADIANCE_AW_ALPHA         0.25       weight of the newest verify in a request's estimate
  RADIANCE_AW_PRIOR         2.0        pseudo-positions pulling q toward the pool
  RADIANCE_AW_COST_A        29.9       ms, T(M) = a + b*M  (prod: ~33 ms at M 8, ~55 ms at M 64, 210 W)
  RADIANCE_AW_COST_B        0.393      ms per verify row
  RADIANCE_AW_COST_TABLE    ""         optional "8:33,16:38,64:55" piecewise-linear (overrides a/b)
  RADIANCE_AW_MIN_ROWS      32         unshrunk row total below which nothing is shrunk
  RADIANCE_AW_QUANT         8          perseq: total rows restricted to multiples (0 = free)
  RADIANCE_AW_VARLEN_GRAPH  1          perseq: capture varlen FULL decode graphs (0 = per-request
                                       widths run PIECEWISE, which is what ggz14's dynwidth did)
  RADIANCE_AW_GRAPH_MIN_REQS 4         uniform: smallest batch (requests) that gets tier graphs
  RADIANCE_AW_LOG_EVERY     500        log a counter line every N decisions (0 = never)
"""
import math
import os
import sys

_MODES = ("0", "uniform", "perseq")


def _env_float(name, default, lo, hi):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        x = float(v)
        if lo <= x <= hi:
            return x
    except ValueError:
        pass
    print(f"[radiance-aw] {name}={v!r} is not a number in [{lo}, {hi}]: keeping {default}", file=sys.stderr)
    return default


def _env_int(name, default, lo, hi):
    return int(_env_float(name, default, lo, hi))


def mode():
    m = (os.environ.get("RADIANCE_ADAPTIVE_WIDTH", "0") or "0").strip().lower()
    if m in ("", "off", "false"):
        m = "0"
    if m not in _MODES:
        print(f"[radiance-aw] RADIANCE_ADAPTIVE_WIDTH={m!r} not in {_MODES}: off", file=sys.stderr)
        return "0"
    return m


def lens_from_env():
    v = os.environ.get("RADIANCE_AW_LENS", "1,3,5,7")
    try:
        out = sorted({int(x) for x in v.split(",") if x.strip()})
        if out and out[0] >= 1:
            return out
    except ValueError:
        pass
    print(f"[radiance-aw] RADIANCE_AW_LENS={v!r} invalid: using 1,3,5,7", file=sys.stderr)
    return [1, 3, 5, 7]


def varlen_graph_enabled():
    return mode() == "perseq" and os.environ.get("RADIANCE_AW_VARLEN_GRAPH", "1") != "0"


def tier_widths(depth):
    """Verify widths (draft tokens) below `depth` that uniform mode may pick; the graph builder
    captures one uniform-decode graph set per width + 1 query length."""
    if mode() != "uniform":
        return []
    return [w for w in lens_from_env() if 1 <= w < depth]


class AdaptiveWidth:
    def __init__(self, depth):
        self.mode = mode()
        self.depth = int(depth)
        self.alpha = _env_float("RADIANCE_AW_ALPHA", 0.25, 0.01, 1.0)
        self.prior = _env_float("RADIANCE_AW_PRIOR", 2.0, 0.0, 1000.0)
        self.cost_a = _env_float("RADIANCE_AW_COST_A", 29.9, 0.0, 1e6)
        self.cost_b = _env_float("RADIANCE_AW_COST_B", 0.393, 0.0, 1e6)
        self.min_rows = _env_int("RADIANCE_AW_MIN_ROWS", 32, 0, 1 << 20)
        self.quant = _env_int("RADIANCE_AW_QUANT", 8, 0, 1 << 20)
        self.log_every = _env_int("RADIANCE_AW_LOG_EVERY", 500, 0, 1 << 30)
        self.table = self._parse_table(os.environ.get("RADIANCE_AW_COST_TABLE", ""))
        # the effective set: configured lengths below the depth, then the depth itself
        self.lens = [w for w in lens_from_env() if 1 <= w < self.depth] + [self.depth]
        self.pool_hits = 0.0
        self.pool_seen = 0.0
        # counters
        self.n_dec = 0          # decisions with >= 1 candidate
        self.n_short = 0        # decisions that shortened at least one request
        self.rows_full = 0
        self.rows_chosen = 0
        self.hist = {}          # chosen width -> request count

    @staticmethod
    def _parse_table(s):
        if not s:
            return None
        try:
            pts = []
            for item in s.split(","):
                m, ms = item.split(":")
                pts.append((float(m), float(ms)))
            pts.sort()
            if len(pts) >= 2 and all(b[0] > a[0] for a, b in zip(pts, pts[1:])):
                return pts
        except ValueError:
            pass
        print(f"[radiance-aw] RADIANCE_AW_COST_TABLE={s!r} invalid: using a + b*M", file=sys.stderr)
        return None

    # ---- estimate -----------------------------------------------------------------------
    def observe(self, req, drafted, accepted):
        if drafted <= 0:
            return
        accepted = max(0, min(accepted, drafted))
        seen = accepted + (1.0 if accepted < drafted else 0.0)
        st = getattr(req, "_rad_aw", None)
        if st is None:
            st = req._rad_aw = [0.0, 0.0]
        g = 1.0 - self.alpha
        st[0] = g * st[0] + accepted
        st[1] = g * st[1] + seen
        self.pool_hits = 0.99 * self.pool_hits + accepted      # slow: only a prior for new requests
        self.pool_seen = 0.99 * self.pool_seen + seen

    def q_pool(self):
        return self.pool_hits / self.pool_seen if self.pool_seen > 0.0 else 0.75

    def q_of(self, req):
        qp = self.q_pool()
        st = getattr(req, "_rad_aw", None)
        if st is None:
            return qp
        d = st[1] + self.prior
        if d <= 0.0:
            return qp
        return min(1.0, max(0.0, (st[0] + self.prior * qp) / d))

    # ---- cost ---------------------------------------------------------------------------
    def cost_ms(self, rows):
        if self.table:
            pts = self.table
            i = 1
            while i < len(pts) - 1 and rows > pts[i][0]:
                i += 1
            (x0, y0), (x1, y1) = pts[i - 1], pts[i]
            t = y0 + (y1 - y0) * (rows - x0) / (x1 - x0)
        else:
            t = self.cost_a + self.cost_b * rows
        return t if t > 1e-3 else 1e-3

    @staticmethod
    def _chunk(q, lo, hi):
        """sum_{j=lo+1..hi} q^j"""
        v = q ** lo
        s = 0.0
        for _ in range(lo + 1, hi + 1):
            v *= q
            s += v
        return s

    # ---- decision -----------------------------------------------------------------------
    def choose(self, qs, ns, base_rows=0):
        """Per-request widths for candidates with acceptance estimates `qs` and `ns` drafts
        available each. perseq: marginal-gain greedy, rows quantised. uniform: one common width.
        Returns the list of widths (1..n_i)."""
        n = len(qs)
        full_rows = base_rows + sum(1 + k for k in ns)
        if (self.mode == "0" or n == 0 or len(self.lens) < 2 or full_rows < self.min_rows):
            return list(ns)
        # ladder of request i: the set's lengths below n_i, then n_i itself
        ladders = [[w for w in self.lens if w < ns[i]] + [ns[i]] for i in range(n)]
        if self.mode == "uniform":
            return self._choose_uniform(qs, ns, ladders, base_rows)
        return self._choose_perseq(qs, ns, ladders, base_rows, full_rows)

    def _choose_uniform(self, qs, ns, ladders, base_rows):
        n = len(qs)
        top = max(ns)
        best_w, best_f = top, -1.0
        for w in self.lens:
            ks = [min(w, ns[i]) for i in range(n)]
            rows = base_rows + sum(1 + k for k in ks)
            toks = sum(1.0 + self._chunk(qs[i], 0, ks[i]) for i in range(n))
            f = toks / self.cost_ms(rows)
            if f >= best_f:                 # ties go to the wider (later) width
                best_f, best_w = f, w
        return [min(best_w, ns[i]) for i in range(n)]

    def _choose_perseq(self, qs, ns, ladders, base_rows, full_rows):
        n = len(qs)
        lvl = [0] * n
        rows = base_rows
        toks = 0.0
        for i in range(n):
            k = ladders[i][0]
            rows += 1 + k
            toks += 1.0 + self._chunk(qs[i], 0, k)

        def admissible(r, full):
            return full or self.quant <= 1 or r % self.quant == 0

        path = []
        best_j, best_f = -1, 0.0
        f = toks / self.cost_ms(rows)
        if admissible(rows, rows == full_rows):
            best_j, best_f = 0, f
        while True:
            bi, br = -1, -1.0
            for i in range(n):
                l = lvl[i]
                if l >= len(ladders[i]) - 1:
                    continue
                lo, hi = ladders[i][l], ladders[i][l + 1]
                r = self._chunk(qs[i], lo, hi) / (hi - lo)
                if r > br:                  # strict: ties keep the lower index
                    br, bi = r, i
            if bi < 0:
                break
            l = lvl[bi]
            lo, hi = ladders[bi][l], ladders[bi][l + 1]
            rows += hi - lo
            toks += self._chunk(qs[bi], lo, hi)
            lvl[bi] = l + 1
            path.append(bi)
            f = toks / self.cost_ms(rows)
            if admissible(rows, rows == full_rows) and (best_j < 0 or f >= best_f):
                best_j, best_f = len(path), f
        if best_j < 0:
            best_j = len(path)
        lvl = [0] * n
        for j in range(best_j):
            lvl[path[j]] += 1
        return [ladders[i][lvl[i]] for i in range(n)]

    def decide(self, reqs, ns):
        """Widths for `reqs` (request objects, ns[i] drafts available each); updates counters."""
        qs = [self.q_of(r) for r in reqs]
        ks = self.choose(qs, ns)
        if reqs:
            self.n_dec += 1
            full = sum(1 + k for k in ns)
            got = sum(1 + k for k in ks)
            self.rows_full += full
            self.rows_chosen += got
            if got < full:
                self.n_short += 1
            for k in ks:
                self.hist[k] = self.hist.get(k, 0) + 1
            if self.log_every and self.n_dec % self.log_every == 0:
                self.log()
        return ks

    def log(self):
        n = max(self.n_dec, 1)
        hist = " ".join(f"k{k}={c}" for k, c in sorted(self.hist.items()))
        print(f"[radiance-aw] mode={self.mode} decisions={self.n_dec} shortened={self.n_short} "
              f"({100.0 * self.n_short / n:.1f}%) rows/step full={self.rows_full / n:.1f} "
              f"chosen={self.rows_chosen / n:.1f} saved={(self.rows_full - self.rows_chosen) / n:.2f} "
              f"q_pool={self.q_pool():.3f} widths: {hist}", flush=True)
