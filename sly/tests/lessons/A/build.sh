#!/bin/bash
# Compile /root/lessons/A/radiance_mxfp4_fp8.hip -> .so inside the image (no GPU needed). Log: build.log
set -e
docker run --rm -e HIP_VISIBLE_DEVICES=-1 -v /root/lessons/A:/w --entrypoint bash ${IMAGE:-vllm-sly-radiance:0.7.0-rocm10.0} -c \
  'cd /w && hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage $(python -m pybind11 --includes) radiance_mxfp4_fp8.hip -o radiance_mxfp4_fp8.so' > /root/lessons/A/build.log 2>&1
ls -la /root/lessons/A/radiance_mxfp4_fp8.so
