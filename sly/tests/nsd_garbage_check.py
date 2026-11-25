#!/usr/bin/env python3
"""NSD must never accept a slot the standard rule rejects outright: greedy rows and -1 placeholders.

Not collected by pytest (no test_ prefix); sly/tests/test_nsd_cpu.py starts it with TRITON_INTERPRET=1.

    TRITON_INTERPRET=1 python3 sly/tests/nsd_garbage_check.py <patched rejection_sampler_utils.py>

The nucleus-mass kernel skips greedy rows and -1 slots, so their entries in the nsd buffer are never
written. The rejection kernel reads them anyway. The spy lets the real kernel run, then overwrites exactly
those entries with 0.0 (worst case: 0 <= theta, "in the nucleus"), as an uninitialised buffer may hold.
  (a) greedy rows: exactly the argmax rule (first mismatch rejects, target argmax is stored there)
  (b) -1 slots: verification ends there for every row
  (c) the buffer handed to the mass kernel is initialised so that no entry passes the theta test
Fix 1 (gate in the rejection kernel) is what (a)/(b) need; fix 2 (buffer init) is what (c) needs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsd_kernel_check as base  # noqa: E402  (sets TRITON_INTERPRET, stubs vllm)
import torch  # noqa: E402

THETA = 0.99
N, K, VOCAB = 8, 4, 20000


class _Spy:
    """Wraps the nucleus-mass kernel: snapshots the buffer it is handed, then poisons the untouched entries."""

    def __init__(self, real, temps, draft):
        self.real, self.temps, self.draft, self.init = real, temps, draft, None

    def __getitem__(self, grid):
        launch = self.real[grid]

        def call(buf, *args, **kw):
            self.init = buf.clone()
            launch(buf, *args, **kw)
            view = buf.view(N, K + 1, -1)
            for r in range(N):
                for s in range(K):
                    if self.temps[r] == 0.0 or int(self.draft[r, s]) < 0:
                        view[r, s] = 0.0
        return call


def main(patched_path):
    gen = torch.Generator().manual_seed(5)
    temps = torch.tensor([0.0] * 6 + [1.0, 1.0])
    logits = torch.randn(N, K + 1, VOCAB, generator=gen) * 2.5
    top = logits.argmax(-1)                                   # [N, K+1]
    wrong = (top + 1) % VOCAB
    draft = top[:, :K].clone()
    draft[1, 1] = wrong[1, 1]                                 # greedy mismatch, later slots match again
    draft[2, 0] = -1                                          # greedy, masked first slot
    draft[3, 1] = -1                                          # greedy, masked middle slot
    draft[4, :] = wrong[4, :K]                                # greedy, everything wrong
    draft[5, 2] = -1                                          # greedy, masked late slot
    draft[7, 1] = -1                                          # sampled, masked slot (draft[6] all argmax)
    seed = torch.randint(1, 2 ** 62, (N,), generator=gen, dtype=torch.int64)

    mod = base.load(patched_path, "rsu_garbage", RADIANCE_NSD=1, RADIANCE_NSD_THETA=THETA, RADIANCE_CACTUS_DELTA=0)
    spy = _Spy(mod._nsd_nucleus_mass_kernel, temps, draft)
    mod._nsd_nucleus_mass_kernel = spy
    out, num = base.run(mod, temps, logits, draft, seed, K)

    for r in range(N):
        d = draft[r].tolist()
        end = d.index(-1) if -1 in d else K                   # verification ends at the first placeholder
        if temps[r] == 0.0:
            n = 0
            while n < end and d[n] == int(top[r, n]):
                n += 1
            assert int(num[r]) == n + 1, f"(a)/(b) greedy row {r}: accepted {int(num[r]) - 1}, argmax rule says {n}"
            assert out[r, :n].tolist() == d[:n] and int(out[r, n]) == int(top[r, n]), f"greedy row {r} tokens"
        else:
            assert int(num[r]) - 1 <= end, f"(b) sampled row {r}: accepted {int(num[r]) - 1} past the placeholder at {end}"

    assert spy.init is not None, "nucleus-mass kernel was not launched"
    buf = spy.init.view(N, K + 1, -1)
    for r in range(N):
        for s in range(K):
            if temps[r] == 0.0 or int(draft[r, s]) < 0:
                assert float(buf[r, s].sum()) > THETA, f"(c) uninitialised entry ({r},{s}) passes the theta test: {buf[r, s]}"
    print("nsd garbage check OK")


if __name__ == "__main__":
    main(sys.argv[1])
