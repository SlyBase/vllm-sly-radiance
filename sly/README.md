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
| `mxfp4/radiance_mxfp4_fp8.hip` | Hand-geschriebener fp8-WMMA-W4A8-GEMM-Kernel (Prefill), von [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4) |
| `mxfp4-configs/` | Getunete MXFP4-GEMM-Configs (JSON, wie `fp8-configs/`) — noch leer, folgt nach erstem echten GPU-Tuning-Lauf |

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
