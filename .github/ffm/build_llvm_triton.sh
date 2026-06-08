#!/bin/bash
# Build AMD's custom LLVM, then build Triton against it.
set -e

LLVM_REPO=${LLVM_REPO:-/root/llvm-project}
TRITON_REPO=${TRITON_REPO:-/root/triton}

[ -d "$LLVM_REPO" ] || { echo "Error: LLVM repo not found at $LLVM_REPO"; exit 1; }
[ -d "$TRITON_REPO" ] || { echo "Error: Triton repo not found at $TRITON_REPO"; exit 1; }

echo "=== Step 1: Build LLVM ==="
cd "$LLVM_REPO"
rm -rf build && mkdir build && cd build
cmake -GNinja \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=True \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86;NVPTX;" \
    -DLLVM_ENABLE_Z3_SOLVER=OFF \
    -DLLVM_ENABLE_PROJECTS="clang;mlir;lld;" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DLLVM_APPEND_VC_REV=OFF \
    ../llvm/
ninja

echo "=== Step 2: Build Triton against custom LLVM ==="
export LLVM_DIR=${LLVM_REPO}/build
export PATH=${LLVM_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${LLVM_DIR}/lib:${LD_LIBRARY_PATH}

cd "$TRITON_REPO"
rm -rf build

export PIP_BREAK_SYSTEM_PACKAGES=1
python3 -m pip uninstall -y triton || true

export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_BUILD_WITH_CCACHE=true
export DEBUG=true
export TRITON_ALWAYS_COMPILE=1
export LLVM_BUILD_DIR=${LLVM_DIR}
export LLVM_INCLUDE_DIRS=${LLVM_DIR}/include
export LLVM_LIBRARY_DIR=${LLVM_DIR}/lib
export LLVM_SYSPATH=${LLVM_DIR}

python3 -m pip install -e .

echo "=== Done ==="
echo "LLVM built at ${LLVM_REPO}/build"
echo "Triton installed from ${TRITON_REPO}"
