# GDN `chunk_scan` v2: killing the four spills by moving one live range

Status 2026-10-20, branch `exp/gdn-scan2`, nothing pushed, **nothing run on the GPU**. Knob
`RADIANCE_GDN_SCAN_V2`, default `0` = stock.

## Why this slice, and not the other two candidates

The w3 slice (LDS diet to 43,520 B, 3 workgroups per WGP) was measured at T = 4089, H = 48 and came
out 9 % **slower** while bit-identical. That result prices the two levers the spec had left:

* **More waves do not pay.** w3 raised the occupancy from 4 to 6 waves/SIMD and lost. The kernel is
  therefore not occupancy-limited at 4 waves/SIMD; it is limited per workgroup. Any slice that buys
  speed with waves has to overcome that measurement first.
* **A barrier per chunk is expensive.** w3 added exactly one (`(1b)`) and the 9 % loss is at least
  partly its. Which is why the barrier count is the *worst* place to look: the stock schedule has
  four, and the hazard model of w3 (`sly/gdn/tests/test_chunk_scan_w3_hazard.py`) shows each one is
  load-bearing for this tiling -- `(2)` publishes W/sA, `(3)` publishes V', `(4)` separates the state
  loop's readers of `s.k`/`s.D` from the next chunk's staging writes. Removing one needs a different
  tiling, i.e. exactly the restructure w3 already lost with.

The prefetch candidate is likewise spent: the stock kernel already issues the next chunk's six
staging loads a whole chunk ahead, and the in-loop variant (`PFIN`) was measured 5 % slower.

What was left unexamined was the kernel's own statement of distress: at 4 waves/SIMD the VGPR ceiling
is 256, the kernel sits **on** it, and the compiler spills four registers (20 B/lane). **Hypothesis
(a): remove the spills by shortening the live range that causes them, and change nothing else.**
That is the one lever that cannot cost waves, cannot cost LDS, cannot add a barrier and cannot move a
single arithmetic instruction.

## The live range

At `WIDE` a lane stages eight shorts per load: `kb[4] + vb[2]` = 6 `uint4` = **24 VGPRs**. The stock
unit issues that prefetch immediately after barrier `(1)` and consumes it at the top of the next
chunk, so those 24 registers are live across the entire body -- including the main WMMA loop, which
already peaks at three accumulator sets (`O`, `u`, `p`, 48 VGPRs) plus five operand fragments.

v2 moves the issue point into the chunk tail. The addresses, the loads, their order relative to each
other and every consumer are unchanged; only the distance between the issue and the WMMA-dense
region changes. Bit-identical by construction, not by argument.

`PLATE` selects the point: `0` = stock (control), `1` = after the main loop (the default: the
prefetch still spans W/S, V' and O plus the state update), `2` = just before the state update (the
shortest live range, with only the state loop and the writeback left as the latency window).

## Resource report (hipcc, gfx1201, libr4d pin a3e4833 + `r4d_extras_rx10.patch`, CPU compile)

`ci/gdn_scan_resources.sh` / `ci/tests/fixtures/gdn_scan_v2.remark.txt`:

| Build | VGPR | spills | scratch B/lane | LDS B | waves/SIMD | SGPR |
|---|--:|--:|--:|--:|--:|--:|
| stock `chunk_scan` (`-mcumode`) | 256 | 4 | 20 | 60,928 | 4 | 69 |
| v2, `PLATE=0` (unit split only) | 256 | 4 | 20 | 60,928 | 4 | 66 |
| **v2, `PLATE=1` (default)** | **205** | **0** | **0** | **60,928** | **4** | **66** |
| v2, `PLATE=2` | 188 | 0 | 0 | 60,928 | 4 | 66 |
| v2, `PLATE=1`, `LDSO2=1` | 208 | 0 | 0 | 60,928 | 4 | 66 |
| v2, `PLATE=2`, `LDSO2=1`, `APRE=1` | 256 | 2 | 12 | 60,928 | 4 | 66 |

`PLATE=0` compiling to the stock figures is the control: the new translation unit and the rename add
no pressure of their own, so the difference is the live range and nothing else.

Static ISA, `PLATE=1` against stock (`amdclang++ --cuda-device-only -S`, count of real ISA lines:
1382 -> 1384): `v_wmma` 80/80, `global_load` 46/46, `global_store` 10/10, `s_barrier` 10/10,
`v_exp` 3/3, `v_min_u32` 12/12, and the four `scratch_*` ops are gone (4 -> 0). What does move is
where the freed registers let the scheduler choose differently: the VOPD pairing (`v_dual_mul_f32`
66 -> 62, `v_mul_f32` 25 -> 34, i.e. one multiply rematerialised rather than kept alive), the address
reconstruction (`v_lshlrev_b32` 29 -> 31, `v_add_nc_u32` 174 -> 173) and the waitcnt shape
(`s_wait_loadcnt` 18 -> 19). None of that is an arithmetic step of the kernel: the source computes the
same expressions on the same values, and `-ffp-contract=off` keeps every mul and add separate, so the
per-element result is unchanged. The proof of that is the GPU DIGEST gate, not this listing.

Honest reading of the numbers: the spill traffic itself is not the prize. The store/reload pair sits
**outside** the chunk loop (one 2 x b64 store, one reload per launch = ~5 KB per launch), so removing
it is worth almost nothing on its own. The finding that matters is the headroom: 51 VGPRs of it at
`PLATE=1`, and 68 at `PLATE=2`, in a kernel that previously could not keep one more address alive.
`LDSO2` -- the opaque byte-offset addressing for the V'/O/state loops that the stock comment rejects
because it spilled 146 registers at NW=8 -- now compiles clean (208 VGPR, 0 spills). It is left at 0
here because its static effect is small (1142 -> 1137 instructions, 0.4 %), and a 0.4 % static change
is not worth an unmeasured behaviour change; it is listed as an arm for the GPU window instead.

**The 379.8 -> ~280 ms target is not demonstrated by this slice.** A 26 % gap cannot be closed by
four spills, and nothing here has been timed. What is demonstrated is that the spill wall is gone and
that the kernel now has room to be scheduled -- the timing question is the GPU gate below, and it is
expected to come out well short of 0.92 unless the headroom translates.

## Gates

| Gate | Where | Pass condition |
|---|---|---|
| reference math unchanged | `sly/gdn/tests/test_chunk_scan_ref.py` (CPU) | fp64 chunked == fp64 naive, relRMSE <= 1e-10 |
| resource budget | `ci/tests/test_gdn_v2_budget.py` (fixture) | VGPR <= 256, **spills 0**, scratch 0, LDS 60,928, 4 waves/SIMD |
| patch wiring | `ci/tests/test_gdn_scan_v2_patch.py` (CPU; applies the patch to a pin copy when `R4D_SRC` is set) | applies clean, registry row with `variant="v2"` first, module binding, build.sh unit |
| knob | `ci/tests/test_gdn_scan_v2_select.py` (CPU) | unset/0 = stock and no `variant` query; 1 = v2; 1 without v2 = stock + one message |
| **GPU** | `sly/gdn/tests/gpu_chunk_scan_v2.py` **[GPU window]** | DIGEST (`o`, `ht`) == stock for T in {64, 65, 513, 880, 1514, 1760, 4089} x N in {1, 2}; accuracy vs fp64 o <= 5e-3 / state <= 4e-3; timing T = 4089 H = 48 ABBA median 20 |

Timing verdict: **GO** if ratio <= 0.92, **ABORT** if > 0.95 (unchanged from the G1 gate). Run it
once per build of the three `PLATE` arms; the script prints the ratio of whichever arm the loaded
`r4d.so` carries.

## Build and run (GPU window)

```bash
# on the GPU host, pin checkout with extras applied
git apply sly/r4d/r4d_gdn_scan_v2.patch          # after r4d_extras_rx10.patch
cp sly/r4d/r4d_gdn_chunk_scan_k128_v128_c64_bf16_v2.hip .
./build.sh                                       # CPU only, no --device
# then, in the release image with that r4d.so bind-mounted:
python3 -u sly/gdn/tests/gpu_chunk_scan_v2.py
# and the same with RADIANCE_GDN_SCAN_V2=1 for the engine arm
```
