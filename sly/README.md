# sly/ — SlyBase-Ergänzungen zu vllm-radiance

Dieses Verzeichnis enthält alle SlyBase-eigenen Patches und Configs on top of
[StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance).
Ziel: RDNA4/gfx1201-MXFP4-Support für `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` auf
AMD Radeon AI PRO R9700.

## Inhalt

| Datei | Zweck |
|---|---|
| `patch_quark_mxfp4.py` | AITER-Triton-MXFP4-Gate für gfx1201 (5 Hunks) + Registrierung des HIP-Kernel-Plugins |
| `mxfp4/radiance_mxfp4.py` | `RadianceMxfp4W4A8LinearKernel`-Plugin (dispatcht große M auf den HIP-Kernel, sonst AITER) |
| `mxfp4/radiance_mxfp4_fp8.hip` | Hand-geschriebener fp8-WMMA-W4A8-GEMM-Kernel (Prefill), von [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4). **2026-09-15**: `DEC_MAX_N` 32768 → 36864 — Qwen3.8-27B (TP=1) hat einen fusionierten gate_up mit N = 34816, den die alte Grenze still in den Folded-(Prefill-)Kernel schickte (64 Calls/Step à ~420 µs statt ~190 µs = 30 von 72 ms je DFlash-Step); Scratch + Block-Counter in `radiance_mxfp4.py` passend vergrößert |
| `mxfp4-configs/` | MXFP4-GEMM-Configs für AITERs Config-Lookup (JSON, wie `fp8-configs/`) — `gfx1201/.../gemm_afp4wfp4/DEFAULT.json` war ursprünglich AITERs eigener gfx950/gfx1250-Default 1:1 übernommen; AITER hat für diese GEMM-Familie **kein** gfx1201-Tuning und bricht ohne Config-Datei hart mit `AssertionError` ab (kein eingebauter Fallback). **2026-09-14 (Task #5, Build #8 First-Request-Crash)**: dieser Default crashte den EngineCore beim allerersten echten Request mit `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 67584-100352, Hardware limit: 65536` — gfx950/gfx1250 haben deutlich mehr LDS pro CU (`_LDS_CAP_BYTES` in aiter: gfx1250=327680, gfx950=163840; **gfx1201 fehlt in dieser Map komplett**), gfx1201/RDNA4 hat nur 64KiB. Gefixt per Brute-Force-Probe direkt auf der R9700 (echter `gemm_afp4wfp4()`-Launch pro M-Bucket, bei `OutOfResources` `num_stages` schrittweise runter, bis der Launch durchläuft): alle 8 Buckets brauchten nur `num_stages` 3→2 (bzw. 2→1 für `any`) reduziert, `BLOCK_SIZE_*` unverändert. Das ist ein reiner Korrektheits-Fix (macht den Kernel überhaupt lauffähig) — **echtes do_bench-Perf-Tuning der Blockgrößen selbst ist weiterhin offene Task-#7-Arbeit** |
| `patch_short_prefill.py` | GDN-Fix: 1-Token-Prefill wird nicht mehr fälschlich als Decode klassifiziert (Subagent E, ursprünglich vllm5-Bind-Mount-Patch für 0.28.0, hier verbatim gegen 0.29.0 verifiziert) |
| `patch_dflash_w4_packed.py` | DFlash-Drafter als compressed-tensors-W4A16-Checkpoint (`syvai/Qwen3.8-27B-DFlash2-W4A16`): `qkv_proj` hat `weight_packed`, kein rohes `.weight` → wie der fp8-Fall deferred + über den eigenen Forward dequantisiert (portiert vom vllm5-Bind-Mount-Patch 0.27.1, verifiziert 0.29.0 / vllm7 2026-09-15) |
| `patch_gdn_nonspec_mask.py` | `patch_gdn_metadata`s numpy-Pfad lässt `non_spec_sequence_masks_cpu` unbelegt → `UnboundLocalError` beim Engine-Init, sobald `--speculative-config` gesetzt ist; Einzeiler, der den Tensor aus der numpy-Maske rekonstruiert |

**Kernel-Entscheidung (Subagent C1, abgeschlossen)**: kein Entweder-Oder — beide
Pfade werden gebraucht, geschichtet:

1. **Pflicht-Basis**: `patch_quark_mxfp4.py`-Hunks 2–5 schalten AITERs
   Triton-`gemm_afp4wfp4` (natives MXFP4-W4A4) auf gfx1201 frei
   (`RADIANCE_MXFP4=1`). Ohne das fällt jede MXFP4-Linear-Layer auf
   `EmulationMxfp4LinearKernel` (BF16-Dequant+F.linear) zurück.
2. **Optimierung obendrauf**: `RadianceMxfp4W4A8LinearKernel`
   (`mxfp4/radiance_mxfp4.py` + `.hip`-Kernel, `RADIANCE_MXFP4_W4A8=1`) wird
   per Hunk 1 an den Kopf der ROCm-Kernel-Liste gesetzt und übernimmt große-M
   (Prefill-)Shapes mit fp8-Aktivierungen (1.6–1.9x schneller als der getunte
   AITER-Pfad, laut ggz14s `PERFORMANCE.md`); `can_implement()`/`is_supported()`
   lehnen für alles andere ab und die Anfrage fällt zurück auf (1).

Alle 5 Hunks in `patch_quark_mxfp4.py` wurden gegen den echten vLLM-0.29.0-Quellcode
(Tag `v0.29.0`) getestet: Anchor-Match + `ast.parse()` + idempotenter
Zweitlauf (NOOP) + `py_compile`, alles grün. Details/Begründung je Hunk im
Docstring der Datei. Zwei der fünf Hunks (4 + 5) sind **neu gegenüber
ggz14s Original** — vLLM hat zwischen 0.27 und 0.29 zwei zusätzliche
CDNA-only-Gates um den AITER-Custom-Op gezogen, die ggz14s 0.27.1-Ziel noch
nicht kannte.

`patch_gfx1201.py` (Top-Level, von `StillDeadcode/vllm-radiance` geerbt, byte-
identisch mit ggz14s Version) ist bereits im Baum und wurde ebenfalls gegen
vLLM 0.29.0 + Triton 3.8.0 verifiziert — alle vier Anchors (gcn-arch-Env,
AITER-CDNA-Gate, Triton-`HIPDriver.is_active`, AITER-Sampler-Gate) matchen
verbatim, keine Änderung nötig.

**`patch_unified_attention_lds.py` (Top-Level, LDS-Fix + bf16-Tuning für gfx1201)
auf aiter 0.1.21.post2 portiert**: AITER hat zwischen der alten Version (die
dieser Patch ursprünglich patchte) und `0.1.21.post2` die komplette
Config-Auswahl in `unified_attention.py` umgebaut — `select_3d_config`/
`select_2d_config` (Python-elif-Ketten) sind weg, ersetzt durch
tabellengetriebene JSON-Configs (`get_unified_attention_config()` in
`unified_attention_utils.py`, geladen per `json.load()` — also reine Daten,
nicht Python, `_patchlib.apply()`/`ast.parse()` kann dort also nicht ansetzen).
Der Patch wurde komplett neu geschrieben (6 Hunks statt 3) gegen die reale,
per `raw.githubusercontent.com` geladene `0.1.21.post2`-Quelle:
- **LDS-Fit-Klemme** (Korrektheit, unbedingt, 2D **und** 3D) sitzt jetzt in
  `_unified_attention_2d_triton()`/`_unified_attention_3d_triton()`, direkt
  nach dem Config-Lookup, vor dem jeweiligen Kernel-Launch.
- **Konsistenz-Problem gelöst**: `kernel_unified_attention_3d` und
  `reduce_segments` leiten aus demselben `TILE_SIZE` unabhängig voneinander
  ihre Segment-Aufteilung ab (`tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS
  * TILE_SIZE)`) — würde man `TILE_SIZE` nur lokal in
  `_unified_attention_3d_triton()` klemmen, liefe `reduce_segments` mit dem
  alten Wert weiter und würde stillschweigend falsche Segmente mergen. Fix:
  `_unified_attention_3d_triton()` gibt sein (geklemmtes/getuntes) `TILE_SIZE`
  jetzt per `return` zurück, der einzige Call-Site in `unified_attention()`
  fängt das ab und reicht denselben Wert an den nachfolgenden
  `_reduce_segments_triton()`-Call weiter.
- **bf16/fp16-3D-Decode-Tuning** (TILE16/warps4/stages2/waves2, plus
  passendes `num_warps=4` im Reduce-Kernel) strukturell an die neue Stelle
  portiert, hinter `DEVICE_ARCH == "gfx1201"` gated (neu ggü. dem alten Patch —
  die alte RDNA-Verzweigung war implizit, die neue Architektur ist arch-
  agnostisch, ein ungegatetes Override hätte andere Archs auf diesem Fork
  mit-getroffen).
Verifiziert (ohne GPU): Anchor-Match (alle 6 Stellen `count==1` gegen die
echte `0.1.21.post2`-Datei), `ast.parse()`, idempotenter Zweitlauf (NOOP),
`py_compile` — alles grün (`patch-verify-e2/` im Scratchpad, nicht Teil des
Commits). **Offen**: ob TILE16/warps4/stages2/waves2 in der neuen
Tabellenstruktur weiterhin do_bench-optimal sind, muss auf echter R9700-
Hardware neu vermessen werden — hier nur strukturell/korrekt an die neue
Stelle portiert, nicht neu getuned.

**Offener Punkt für Subagent B**: der `.hip`-Kernel selbst braucht noch
Build-Wiring im Dockerfile (Compile-Schritt + Extension-Load, analog zu
`radiance_kernels.py`s Pattern für die anderen `.hip`-Kernel in diesem Repo,
z.B. R4D) — reine Python-Seite (Hunk 1 + `radiance_mxfp4.py`) ist fertig,
die kompilierte `.so` fehlt noch.

`RADIANCE_MXFP4_KERNEL`-Env aus der ursprünglichen Plan-Fassung entfällt —
die Auswahl passiert automatisch über die Kernel-Priorität
(`_POSSIBLE_MXFP4_KERNELS[ROCM]`), nicht über einen manuellen Schalter.

## Git-Branching / Upstream-Merge

```
main           ← upstream tracking (mirror von StillDeadcode/vllm-radiance, Codeberg)
  └─ sly/main  ← Integration (alle sly-Patches auf main aufgebaut)
```

Remotes:
- `origin`   = `https://github.com/SlyBase/vllm-sly-radiance.git`
- `upstream` = `https://codeberg.org/StillDeadcode/vllm-radiance.git`

Merge-Prozedur bei Upstream-Updates:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

git checkout sly/main
git rebase main
# Patch-Anchor-Konflikte sind harte Fehler (_patchlib.apply() Uniqueness-Check) —
# jeden betroffenen Patch in sly/ gegen den neuen Anchor-String re-verifizieren
git push origin sly/main --force-with-lease
```

## Build & Push

```bash
docker build --build-arg ROCM_BASE=rocm/dev-ubuntu-24.04:10.0.0-full \
  -t ghcr.io/slybase/vllm-sly-radiance:<VERSION>-rocm<ROCM_VERSION> .
docker push ghcr.io/slybase/vllm-sly-radiance:<VERSION>-rocm<ROCM_VERSION>
```

`<VERSION>` = Inhalt von `../VERSION` (eigene SemVer, unabhängig von der
upstream radiance-Version). `docker login ghcr.io` muss vorher lokal (auf
LXC 2408) mit einem PAT mit `write:packages`-Scope eingerichtet sein.
