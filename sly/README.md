# sly/ — SlyBase-Ergänzungen zu vllm-radiance

Dieses Verzeichnis enthält alle SlyBase-eigenen Patches und Configs on top of
[StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance).
Ziel: RDNA4/gfx1201-MXFP4-Support für `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` auf
AMD Radeon AI PRO R9700.

## Inhalt

| Datei | Zweck |
|---|---|
| `patch_aiter_gfx1201_mxfp4.py` | AITER `_ARCH_TO_DEVICE` + MXFP4-Gate-Fix für gfx1201 (nur Triton-Fallback-Pfad) |
| `patch_mxfp4_gemm_tune.py` | GEMM-Tuning-Configs für gfx1201 (nur Triton-Fallback-Pfad) |
| `mxfp4-configs/` | Getunete MXFP4-GEMM-Configs (JSON, wie `fp8-configs/`) |
| `radiance_mxfp4_fp8.hip` (falls portiert) | Primärer MXFP4-W4A8-Kernel, portiert von [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4) |

Kernel-Entscheidung (siehe Plan `groovy-floating-twilight.md`, Subagent C1):
zuerst wird der HIP-Kernel von ggz14 auf die hier gepinnte vLLM-Version
portiert; nur wenn das scheitert, kommt der AITER-Triton-Pfad
(`patch_aiter_gfx1201_mxfp4.py` + `mxfp4-configs/`) zum Einsatz.
Auswahl zur Laufzeit über `RADIANCE_MXFP4_KERNEL=hip|triton`.

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
