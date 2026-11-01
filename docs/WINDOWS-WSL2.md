# Windows (WSL2)

The image runs on Windows 11 in WSL2. WSL2 does not have `/dev/kfd` or `/dev/dri`. ROCm uses `/dev/dxg`
and AMD's [librocdxg](https://github.com/ROCm/librocdxg) to access the GPU. You mount librocdxg from
the host. We tested this procedure on one R9700 with Windows 11 Pro 26200, Ubuntu 24.04 and librocdxg 1.2.2.

**Use Adrenalin 26.8.1 or a later version.** Older drivers, for example 25.20.42.14, stop the GPU during
long compute work in WSL.

## Setup

In Windows, do these steps:

1. Install Adrenalin.
2. Install Ubuntu: `wsl --install -d Ubuntu-24.04`.
3. Put the text that follows in `%UserProfile%\.wslconfig`:

   ```ini
   [wsl2]
   memory=32GB
   swap=0
   ```

4. Run `wsl --shutdown`.

In the distro, do these steps as root:

1. Install [Docker CE](https://docs.docker.com/engine/install/ubuntu/).
2. Install ROCm 7.2.4 and librocdxg with these commands:

```bash
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb \
     https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb \
     https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-amd-smi-lib_1.2.2_amd64.deb
apt-get install -y ./amdgpu-install_7.2.4.70204-1_all.deb
amdgpu-install -y --usecase=rocm --no-dkms
apt-get install -y ./rocdxg-roct_1.2.2_amd64.deb ./rocdxg-amd-smi-lib_1.2.2_amd64.deb
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo | grep gfx1201
```

Download the model and the drafter. The [quickstart](../README.md#quickstart) gives the commands. To get `hf`
on Ubuntu 24.04, do these steps:

1. Run `apt-get install -y pipx`.
2. Run `pipx install huggingface_hub`. This command installs `hf` in `/root/.local/bin`.

The 1.x images use ROCm 10.1. ROCm 10.1 has its own copy of the `/dev/dxg` bridge, `librocdxg.so.1.1.0`.
AMD builds this copy from the rocm-systems tree. In WSL, this copy is much slower than the librocdxg 1.2.2
release. [Performance](#performance) shows the data. Replace the copy with librocdxg 1.2.2.

librocdxg 1.2.2 needs the ROCm 10.0 `libhsa-runtime64`. The ROCm 10.1 library stops with an error when it
loads librocdxg 1.2.2, because librocdxg 1.2.2 does not have `hsaKmtGetDefaultHostGpu`. Copy the ROCm 10.0
library from the `0.4.5-rocm10.0` image. We tested the procedure with the library from this image. Do this step
one time:

```bash
mkdir -p /opt/rocm10.0-hsa
docker run --rm --entrypoint bash -v /opt/rocm10.0-hsa:/out ghcr.io/slybase/vllm-sly-radiance:0.4.5-rocm10.0 \
  -c 'cp -L /opt/rocm/core-10.0/lib/libhsa-runtime64.so.1.21.0 /out/'
```

**This configuration mixes ROCm 10.0 and ROCm 10.1 files. AMD does not support it. Do the check below again
after each change of the image.**

Start the server. This command is the quickstart command with the changes for WSL. It uses the 1.1.1 image,
because 1.1.1 has the split-K fence fix:

```bash
docker run -d --name vllm --restart unless-stopped \
  --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -v /opt/rocm/lib/librocdxg.so.1.2.2:/opt/rocm/core-10.1/lib/librocdxg.so.1.1.0:ro \
  -v /opt/rocm10.0-hsa/libhsa-runtime64.so.1.21.0:/opt/rocm/core-10.1/lib/libhsa-runtime64.so.1.21.0:ro \
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
  -e RADIANCE_MXFP4_WIDE_MAX_M=192 -e RADIANCE_ADAPTIVE_WIDTH=perseq \
  ghcr.io/slybase/vllm-sly-radiance:1.1.1-rocm10.1 \
  --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --quantization quark \
  --max-model-len 262144 --gpu-memory-utilization 0.94 --kv-cache-memory-bytes 13955000000 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config.method dflash \
  --speculative-config.model syvai/Qwen3.8-27B-DFlash2-W4A16 \
  --speculative-config.num_speculative_tokens 7 \
  --speculative-config.draft_sample_method probabilistic \
  --speculative-config.attention_backend TRITON_ATTN \
  --attention-backend R4D_HYBRID \
  --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
  --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]' \
  --enable-prefix-caching --skip-mm-profiling --enable-mm-embeds \
  --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template /opt/qwen-fixed.jinja \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}'
```

Make sure that the two mounts replace the files of the image. The two values in each pair must be the same:

```bash
sha256sum /opt/rocm/lib/librocdxg.so.1.2.2 /opt/rocm10.0-hsa/libhsa-runtime64.so.1.21.0
docker exec vllm sha256sum /opt/rocm/core-10.1/lib/librocdxg.so.1.1.0 \
  /opt/rocm/core-10.1/lib/libhsa-runtime64.so.1.21.0
```

**Keep a `wsl.exe` session open while the server runs.** WSL stops a distro when no `wsl.exe` session connects
to it. This also stops the container. For example, run `wsl -d Ubuntu-24.04 -- sleep infinity`.

## Changes from the Linux command

| Change | Reason |
|---|---|
| `--device=/dev/dxg`, `/usr/lib/wsl` mounted | These are the GPU device and the driver libraries of WSL. |
| librocdxg 1.2.2 and the ROCm 10.0 `libhsa-runtime64` mounted on the files of the image, `HSA_ENABLE_DXG_DETECTION=1` | ROCm uses librocdxg to send work through `/dev/dxg`. The ROCm 10.1 bridge in the image is approximately 2x slower when the load is high. |
| `/opt/rocm-wsl` mounted, `PYTHONPATH`, `LD_LIBRARY_PATH` | This amd-smi operates without amdgpu sysfs. |
| `ROCPROFILER_REGISTER_ENABLED=0` | rocprofiler-register tries to find `/dev/kfd` when it loads. |
| `VLLM_WSL2_ENABLE_PIN_MEMORY=1` | vLLM does not use pinned memory in WSL if you do not set this value. |
| `--gpu-memory-utilization 0.94`, `--kv-cache-memory-bytes 13955000000` | The Windows desktop uses up to 1.7 GiB of VRAM, so the README value 0.98 fails on Windows. The byte value sets the size of the KV pool directly: 385k tokens, the same as 0.4.x. |

## Performance

We measured these values on one R9700 at 210 W, with Windows 11, WSL2 and Adrenalin 26.8.1. We did all
the tests in one session on 2026-10-08. The test conditions are:

- Decode: temperature 0, thinking off, a story of 1,024 tokens. The mean draft acceptance is approximately 2.7.
- Prefill: one prompt of 21k tokens.
- c4 and c8: 4 or 8 requests of 768 tokens at the same time.
- Engine init: the time that vLLM writes in its log line "init engine (profile, create kv cache, warmup
  model)", with warm caches. This time does not include the model load and the start of the container.

Each decode value is the mean of 3 requests. Each prefill value is the mean of 2 prompts. The c4 and c8 values
come from one test each.

| | 0.4.0-rocm10.0 | 1.1.1-rocm10.1 as shipped | 1.1.1-rocm10.1, bridge replaced (above) |
|---|---|---|---|
| decode, 1 stream | 63 tok/s | 47 | **70** |
| c4 / c8 aggregate | 203 / 273 | 109 / 154 | 205 / **322** |
| prefill 21k | 2.3k tok/s | 2.2k | 2.4k |
| engine init, warm caches | 31 s | 185-250 s | **24 s** |

A reverse test shows that the bridge causes the loss. We put only `librocdxg.so.1.1.0` into a fast ROCm 10.0
stack, and all of the loss occurred again. The ROCm 10.1 compiler, HIP and ROCr do not cause the loss.

The driver does not supply AQL hardware queues, so both bridges change AQL packets into PM4 packets on the
CPU. We do not know yet which code difference between the two bridges causes the loss.

**Stop the server when you do not use it.** When the server has no requests, it keeps 4 to 5 CPU cores
busy. WSL does not have GPU interrupts, so the stock ROCm 10.0 ROCr waits in a busy loop.
