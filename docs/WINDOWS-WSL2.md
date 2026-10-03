# Windows (WSL2)

The image runs unchanged on Windows 11 under WSL2. WSL2 has no `/dev/kfd` or `/dev/dri`, so ROCm reaches
the GPU through `/dev/dxg` and AMD's [librocdxg](https://github.com/ROCm/librocdxg), mounted from the host.
Tested on one R9700 with Windows 11 Pro 26200, Ubuntu 24.04 and librocdxg 1.2.2.

**Requires Adrenalin 26.8.1 or later.** Older drivers such as 25.20.42.14 hang the GPU under sustained
compute in WSL.

## Setup

In Windows, install Adrenalin and Ubuntu (`wsl --install -d Ubuntu-24.04`), put this in
`%UserProfile%\.wslconfig` and run `wsl --shutdown`:

```ini
[wsl2]
memory=32GB
swap=0
```

In the distro, as root, install [Docker CE](https://docs.docker.com/engine/install/ubuntu/), then ROCm 7.2.4
and librocdxg:

```bash
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb \
     https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb \
     https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-amd-smi-lib_1.2.2_amd64.deb
apt-get install -y ./amdgpu-install_7.2.4.70204-1_all.deb
amdgpu-install -y --usecase=rocm --no-dkms
apt-get install -y ./rocdxg-roct_1.2.2_amd64.deb ./rocdxg-amd-smi-lib_1.2.2_amd64.deb
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo | grep gfx1201
```

Download the model and drafter as in the [quickstart](../README.md#quickstart). On Ubuntu 24.04, get `hf`
with `apt-get install -y pipx && pipx install huggingface_hub`, which installs it to `/root/.local/bin`.

Start the server with the quickstart command, adapted for WSL:

```bash
docker run -d --name vllm --restart unless-stopped \
  --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -v /opt/rocm/lib/librocdxg.so:/opt/rocm/lib/librocdxg.so:ro \
  -v /opt/rocm/share/rocdxg:/opt/rocm/share/rocdxg:ro \
  -v /opt/rocm-wsl:/opt/rocm-wsl:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm-wsl/lib:/opt/rocm/lib \
  -e PYTHONPATH=/opt/rocm-wsl/share/amd_smi \
  -e HSA_ENABLE_DXG_DETECTION=1 -e ROCPROFILER_REGISTER_ENABLED=0 -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  --security-opt seccomp=unconfined --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm -v triton-cache:/root/.triton -v aiter-cache:/root/.aiter \
  -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=2 \
  -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
  -e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WPERM=1 \
  -e RADIANCE_LMHEAD_INT4=1 -e RADIANCE_FUSED_NORM_QUANT=1 \
  -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 -e RADIANCE_EMBED_BITS=4 \
  ghcr.io/slybase/vllm-sly-radiance:0.4.0-rocm10.0 \
  --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --quantization quark \
  --max-model-len 262144 --gpu-memory-utilization 0.94 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config.method dflash \
  --speculative-config.model syvai/Qwen3.8-27B-DFlash2-W4A16 \
  --speculative-config.num_speculative_tokens 7 \
  --speculative-config.draft_sample_method probabilistic \
  --speculative-config.attention_backend TRITON_ATTN \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --compilation-config.cudagraph_mode FULL_AND_PIECEWISE \
  --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]' \
  --enable-prefix-caching --skip-mm-profiling --enable-mm-embeds \
  --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template /opt/qwen-fixed.jinja \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}'
```

WSL stops a distro when no `wsl.exe` session is attached, which also stops the container. Keep a session
open while the server runs, for example `wsl -d Ubuntu-24.04 -- sleep infinity`.

## Changes from the Linux command

| Change | Reason |
|---|---|
| `--device=/dev/dxg`, `/usr/lib/wsl` mounted | WSL's GPU device and driver libraries |
| librocdxg mounts, `HSA_ENABLE_DXG_DETECTION=1` | ROCm's transport over `/dev/dxg` |
| `/opt/rocm-wsl` mounted, `PYTHONPATH`, `LD_LIBRARY_PATH` | an amd-smi that works without amdgpu sysfs |
| `ROCPROFILER_REGISTER_ENABLED=0` | rocprofiler-register probes `/dev/kfd` at load |
| `VLLM_WSL2_ENABLE_PIN_MEMORY=1` | vLLM disables pinned memory under WSL by default |
| `--gpu-memory-utilization 0.94` | the Windows desktop uses up to 1.7 GiB of VRAM |

## Performance

On 0.4.0 at 210 W, decode and concurrency match the native [reference run](BENCHMARKS.md#reference-run-2026-09-25)
and the first token takes about 30 ms longer. With no requests, the server keeps 4–5 CPU cores busy because
WSL has no GPU interrupts, so stop it when it is not in use.
