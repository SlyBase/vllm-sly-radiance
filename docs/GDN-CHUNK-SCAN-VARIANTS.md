# Opt-in build variants of the GDN chunk scan (s1 / s2 / s3)

`gdn_chunk_scan_k128_v128_c64_bf16` is one kernel that does the WY recompute, the state recurrence and the
output in a single launch, and it is both LDS- and register-bound: 60,928 B of LDS per workgroup and 256
VGPR with a 4-VGPR spill at stock. Three build/structure changes are available as **separate entry points**
in the same `r4d.so`, so each slice can be priced on its own:

| variant | what changes | VGPR | spill | LDS B | waves/SIMD |
|---|---|---|---|---|---|
| stock | `-mcumode`, as shipped | 256 | 4 | 60,928 | 4 |
| `s1` | workgroup-processor mode (the unit is built **without** `-mcumode`) | 256 | 4 | 60,928 | 4 |
| `s2` | `s1` + LDS diet: state mirror shares bytes with sA + W, per-token vectors move into row padding | 256 | 0 | 43,008 | 5 |
| `s3` | `s2` + the next chunk's staging loads issued after barrier (3) instead of after barrier (1) | 189 | 0 | 43,008 | 6 |

Reports from `hipcc -Rpass-analysis=kernel-resource-usage`, gfx1201; the fixtures under
`sly/tests/fixtures/gdn_scan/` are these reports and `sly/gdn/scan_resources.sh` regenerates them. No
`amdgpu_waves_per_eu` hint is used anywhere: 43,008 B of LDS lets three workgroups fit a WGP (3 × 43,008
≤ 128 KB), a workgroup is 4 wave64 so three of them need only 3 waves/SIMD, and `s2` already gives 5. The
189 VGPR of `s3` are the moved prefetch's doing, not a forced register budget — that distinction is what
the fixtures pin.

Since 1.9.0 the image also applies `sly/r4d/r4d_gdn_exact_decay.patch` on top: chunks with a gate span
above 160 take the exact decay form ([libr4d#4](https://github.com/StillDeadcode/libr4d/issues/4)). This
costs registers: `s3` 228 VGPR, still no spill, still 6 waves/SIMD, LDS unchanged. Stock and s1 lose
their spill. The table above is the unpatched build; `sly/tests/fixtures/gdn_scan_exact/` pins the patched
one. Below the threshold the arithmetic is unchanged, so the DIGEST gate below still compares bit for bit.

## Arithmetic is untouched

No arithmetic statement, operand order or rounding site changes; only LDS addresses and the issue point of
loads move. `s2` does change **where** state, sA and W live, which opens LDS hazards, and the schedule
closes them:

* **S aliases sA + W.** S is read only by the main loop and written only by the state update; sA / W are
  written after the main loop and dead by the state update. A new barrier **(1b)**, after the main loop and
  before the first sA / W write, proves every wave is done reading S.
* **SWRITE moves** from the end of a chunk to the top of the next one, behind barrier **(4)**, which also
  proves every wave is done with the previous sA read. The first chunk is unaffected: nothing has been
  written into S's bytes yet, and barrier (1) still separates the write from the first read.
* The four per-token vectors (`ge`, `gv`, `rs`, `bt`) go into k's and D's row padding, which no fragment
  ever reads, so they alias nothing.

`sly/tests/test_gdn_scan_variants_cpu.py` models this schedule (regions, alias map, barriers) and asserts
that it is hazard-free, that removing (1b) or moving SWRITE back before (4) *does* race, and that the
patch carries the barriers the model assumes. Bit-identity against stock on the GPU is the gate below.

## Selecting one

```
RADIANCE_GDN_SCAN_VARIANT=s3        # s1 | s2 | s3; unset = stock
```

`radiance_gdn.py` resolves the name through `r4d.select("gdn_chunk_scan", ..., variant=...)`. A libr4d
without the variants ignores the unknown key and answers with the stock name, and the loader checks that
the answer's name ends in `_s3` (etc.), then warns and keeps stock. Nothing else in the stack changes.

## Gate

```
python3 -u sly/gdn/tests/gpu_chunk_scan_variants.py        # inside an acquired gpu-window, ~3-4 min
```

DIGEST (bit-identity of `o` and `ht` vs stock on 5 shapes incl. a varlen split), STABLE (5 runs per
variant at T=4089) and ABBA-palindromic timing at T=4089 / 1760 / 880, one layer, H=48. Exit 0 = no WRONG;
the per-slice ratios are the measurement. The script does not acquire the window.
