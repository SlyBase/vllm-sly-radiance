#!/usr/bin/env bash
# Strip the ROCm install down to what this image actually runs on: one GPU architecture.
#
# The stock rocm/dev-ubuntu-24.04:*-full tree is ~19 GB, and most of it is device code for GPUs
# this image cannot run on (it is compiled for gfx1201 only) or link-time-only archives. Pruning
# it is worth ~13 GB uncompressed, which is most of the image's download size.
# Steps 6-11 were added for the ROCm 10 layout (TheRock): the per-arch rocBLAS/rocFFT/rocRAND/RCCL
# device code moved out of the libraries into core-<ver>/.kpack/, which steps 1-5 never matched,
# and the toolchain grew a Fortran/MLIR stack. Measured on the 10.0.0 base: another ~3.0 GB
# (the pruned tree 6.0 -> 3.0 GB, apparent size).
#
# This MUST run in a stage whose /opt/rocm is then COPYed into the final image: deleting files in
# a layer on top of the base reclaims nothing, it only records whiteouts.
#
# What is deliberately KEPT:
#   * clang's compiler-rt archives (llvm/lib/clang/**). hipcc links every HIP object against
#     libclang_rt.builtins; AITER JIT-compiles kernels at RUNTIME, so the compiler has to keep
#     working inside the shipped image. Removing these breaks the JIT, not the build.
#   * the device bitcode libraries (amdgcn/**.bc) hipcc needs for the same reason.
#   * every arch-neutral file in the rocBLAS/hipBLASLt library dirs (logic/manifest metadata).
#   * libMIOpen + MIOpenCKGroupedConv_<arch>: libtorch_hip links MIOpen, and the vision encoders'
#     patch-embed convolutions run through it.
#   * lib/llvm/include/{llvm,clang,c++}: only the MLIR and Flang headers go (step 10).
set -eu

GFX="${1:?usage: prune_rocm.sh <gfx-arch>}"
R=/opt/rocm

mb() { local s=0; [ "$#" -gt 0 ] && s=$(du -xc --apparent-size "$@" 2>/dev/null | tail -1 | cut -f1); echo $((s/1024)); }
before=$(du -xs "$R/" 2>/dev/null | cut -f1)

# 1. static archives: link-time only, never loaded at runtime -- except clang's own runtime libs.
mapfile -t archives < <(find "$R/" -name '*.a' -type f ! -path '*/llvm/lib/clang/*')
echo "  -$(mb "${archives[@]}") MB  static archives ($(( ${#archives[@]} )) files, clang runtime kept)"
if [ "${#archives[@]}" -gt 0 ]; then printf '%s\0' "${archives[@]}" | xargs -0 rm -f; fi

# 2. MIOpen convolution kernel libraries for other architectures (~370 MB each).
mapfile -t miopen < <(find "$R/" -name 'libMIOpenCKGroupedConv_gfx*.so' -type f ! -name "*${GFX}*")
echo "  -$(mb "${miopen[@]}") MB  MIOpen conv kernels for other archs ($(( ${#miopen[@]} )) libs)"
if [ "${#miopen[@]}" -gt 0 ]; then printf '%s\0' "${miopen[@]}" | xargs -0 rm -f; fi

# 3. MIOpen perf databases for other architectures.
mapfile -t miodb < <(find "$R/" \( -name '*.kdb*' -o -name '*.db' -o -name '*.db.txt' \) -type f ! -name "*${GFX}*")
echo "  -$(mb "${miodb[@]}") MB  MIOpen perf databases for other archs"
if [ "${#miodb[@]}" -gt 0 ]; then printf '%s\0' "${miodb[@]}" | xargs -0 rm -f; fi

# 4. hipBLASLt kernels: one directory per arch (gfx942 alone is 1.2 GB).
mapfile -t hbl < <(find "$R/lib/hipblaslt/library" -mindepth 1 -maxdepth 1 -type d -name 'gfx*' ! -name "$GFX" 2>/dev/null)
echo "  -$(mb "${hbl[@]}") MB  hipBLASLt kernel dirs for other archs ($(( ${#hbl[@]} )) archs)"
if [ "${#hbl[@]}" -gt 0 ]; then printf '%s\0' "${hbl[@]}" | xargs -0 rm -rf; fi

# 5. rocBLAS Tensile blobs: flat files tagged with the arch in the filename.
mapfile -t rb < <(find "$R/lib/rocblas/library" -type f -name '*gfx*' ! -name "*${GFX}*" 2>/dev/null)
echo "  -$(mb "${rb[@]}") MB  rocBLAS Tensile blobs for other archs ($(( ${#rb[@]} )) files)"
if [ "${#rb[@]}" -gt 0 ]; then printf '%s\0' "${rb[@]}" | xargs -0 rm -f; fi

# 6. kpack archives (ROCm 10): per-arch device code of rocBLAS/rocFFT/rocRAND/RCCL, one file per
#    library and arch, e.g. blas_lib_gfx942.kpack. librocm_kpack opens only the running device's file.
mapfile -t kp < <(find "$R/" -path '*/.kpack/*' -type f -name '*_gfx*.kpack' ! -name "*_${GFX}.kpack")
echo "  -$(mb "${kp[@]}") MB  kpack device-code archives for other archs ($(( ${#kp[@]} )) files)"
if [ "${#kp[@]}" -gt 0 ]; then printf '%s\0' "${kp[@]}" | xargs -0 rm -f; fi

# 7. MIOpen find-dbs and tuning models for other archs: `gfx942130.HIP.fdb.txt`, `gfx90a.tn.model`,
#    `gfx942_*.ktn.model` -- the ROCm 10 names that step 3's *.db / *.db.txt patterns do not match.
#    Only names starting with gfx: an arch-neutral file stays.
mapfile -t miom < <(find "$R/" -path '*/miopen/db/*' -type f -name 'gfx*' \
                      \( -name '*.fdb.txt' -o -name '*.model' \) ! -name "*${GFX}*")
echo "  -$(mb "${miom[@]}") MB  MIOpen find-dbs / tuning models for other archs ($(( ${#miom[@]} )) files)"
if [ "${#miom[@]}" -gt 0 ]; then printf '%s\0' "${miom[@]}" | xargs -0 rm -f; fi

# 8. rocSHMEM device bitcode, one ~10 MB .bc per arch.
mapfile -t shm < <(find "$R/" -type f -name 'librocshmem_device_gfx*.bc' ! -name "*_${GFX}.bc")
echo "  -$(mb "${shm[@]}") MB  rocSHMEM device bitcode for other archs ($(( ${#shm[@]} )) files)"
if [ "${#shm[@]}" -gt 0 ]; then printf '%s\0' "${shm[@]}" | xargs -0 rm -f; fi

# 9. hipSPARSELt kernels: one directory per arch, like hipBLASLt in step 4 (gfx942 alone is 180 MB).
mapfile -t hsl < <(find "$R/lib/hipsparselt/library" -mindepth 1 -maxdepth 1 -type d -name 'gfx*' ! -name "$GFX" 2>/dev/null)
echo "  -$(mb "${hsl[@]}") MB  hipSPARSELt kernel dirs for other archs ($(( ${#hsl[@]} )) archs)"
if [ "${#hsl[@]}" -gt 0 ]; then printf '%s\0' "${hsl[@]}" | xargs -0 rm -rf; fi

# 10. the Fortran/MLIR half of the LLVM toolchain: flang and its front ends, the mlir-* tools,
#     libMLIR plus the libmlir_* runner libs, and their headers. hipcc drives clang + lld only;
#     the check below proves no remaining library or tool links libMLIR.
LL=$(readlink -f "$R/lib/llvm")
mapfile -t fl < <(find "$LL/bin" -maxdepth 1 \( -type f -o -type l \) \( -name 'flang*' -o -name 'amdflang*' \
                     -o -name bbc -o -name 'fir-*' -o -name tco -o -name 'f18-*' -o -name 'mlir-*' \) ; \
                  find "$LL/lib" -maxdepth 1 \( -type f -o -type l \) -iname 'libmlir*'; \
                  find "$LL/include" -mindepth 1 -maxdepth 1 -type d \( -name mlir -o -name flang \))
echo "  -$(mb "${fl[@]}") MB  Flang/MLIR tools, libMLIR and their headers ($(( ${#fl[@]} )) entries)"
if [ "${#fl[@]}" -gt 0 ]; then printf '%s\0' "${fl[@]}" | xargs -0 rm -rf; fi

# 11. ROCm's Python bindings for interpreters this image does not have (the venv is 3.12).
mapfile -t pyb < <(find "$(readlink -f "$R/lib")" -mindepth 1 -maxdepth 1 -type d -name 'python3.*' ! -name python3.12)
echo "  -$(mb "${pyb[@]}") MB  ROCm Python bindings for other interpreters (${pyb[*]##*/})"
if [ "${#pyb[@]}" -gt 0 ]; then printf '%s\0' "${pyb[@]}" | xargs -0 rm -rf; fi

after=$(du -xs "$R/" 2>/dev/null | cut -f1)
echo "  ROCm: $((before/1024)) MB -> $((after/1024)) MB (saved $(( (before-after)/1024 )) MB)"

# Fail loudly if the prune ate something this image needs: the arch's own kernels must survive,
# and hipcc must still be able to compile AND LINK a HIP shared object (the AITER JIT path).
test -f "$R/lib/libMIOpenCKGroupedConv_${GFX}.so" || { echo "FATAL: pruned this arch's MIOpen lib"; exit 1; }
test -d "$R/lib/hipblaslt/library/${GFX}"         || { echo "FATAL: pruned this arch's hipBLASLt kernels"; exit 1; }
ls "$R/lib/rocblas/library/" | grep -q "$GFX"     || { echo "FATAL: pruned this arch's rocBLAS kernels"; exit 1; }
for k in blas rand rccl; do
  find "$R/" -path '*/.kpack/*' -name "${k}_lib_${GFX}.kpack" | grep -q . \
    || { echo "FATAL: pruned this arch's ${k} kpack"; exit 1; }
done
find "$R/" -name "librocshmem_device_${GFX}.bc" | grep -q . || { echo "FATAL: pruned this arch's rocSHMEM bitcode"; exit 1; }
# Nothing left may link the libMLIR that step 10 removed.
while IFS= read -r -d '' f; do
  if "$LL/bin/llvm-readelf" -d "$f" 2>/dev/null | grep -q 'NEEDED.*libMLIR'; then
    echo "FATAL: $f links libMLIR, which step 10 removed"; exit 1
  fi
done < <(find "$(readlink -f "$R/lib")" "$(readlink -f "$R/bin")" "$LL/bin" "$LL/lib" -maxdepth 1 -type f -print0)

cat >/tmp/_jit_probe.hip <<'EOF'
#include <hip/hip_runtime.h>
__global__ void k(float* o) { o[threadIdx.x] = 1.0f; }
extern "C" void launch(float* o) { hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, o); }
EOF
"$R/bin/hipcc" -O3 -fPIC -shared --offload-arch="${GFX}" /tmp/_jit_probe.hip -o /tmp/_jit_probe.so \
  || { echo "FATAL: hipcc can no longer build a HIP shared object -- the AITER runtime JIT would break"; exit 1; }
rm -f /tmp/_jit_probe.hip /tmp/_jit_probe.so
echo "  hipcc still compiles+links HIP shared objects (AITER JIT path intact)"
