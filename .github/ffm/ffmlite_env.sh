#!/bin/bash
pkgroot="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROCM_PATH=$(pip show torch | grep ^Location: | cut -d' ' -f2-)/_rocm_sdk_core
export HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode

export TEST_SUIT_DIR="$pkgroot"
export LD_LIBRARY_PATH=/opt/rocm/lib:"$pkgroot:$pkgroot/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH=/opt/rocm/lib:"$pkgroot:$pkgroot/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export HSA_MODEL_LIB="$pkgroot/lib/libhsakmtmodel.so"
export HSA_MODEL_TOPOLOGY="$pkgroot/topology/mi450"
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export HSA_KMT_MODEL_GPUVM_BASE=0x4000
