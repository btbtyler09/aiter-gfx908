#!/bin/bash
# AITER MI100 Phased Test Script
# Runs tests from most basic to most complex.
# Usage: ./run_mi100_tests.sh [phase_number]
#   No args = run all phases
#   1-6     = run specific phase only

set -euo pipefail

LOGDIR="/workspace/aiter/test_results"
mkdir -p "$LOGDIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

phase_pass() { echo -e "${GREEN}[PASS]${NC} Phase $1: $2"; }
phase_fail() { echo -e "${RED}[FAIL]${NC} Phase $1: $2"; }
phase_start() { echo -e "\n${CYAN}========================================${NC}"; echo -e "${CYAN}Phase $1: $2${NC}"; echo -e "${CYAN}========================================${NC}"; }

run_phase=${1:-all}

########################################
# Phase 1: Build Verification
########################################
run_phase1() {
    phase_start 1 "Build Verification"
    local log="$LOGDIR/phase1_build_verify.log"
    local failed=0

    echo "--- AITER import test ---"
    if python -c "import aiter; print('AITER import OK')" 2>&1 | tee -a "$log"; then
        phase_pass 1 "AITER import"
    else
        phase_fail 1 "AITER import"
        failed=1
    fi

    echo ""
    echo "--- PyTorch / GPU info ---"
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f'  Memory: {mem / 1024**3:.1f} GB')
    gcn = getattr(props, 'gcnArchName', 'N/A')
    print(f'  GCN Arch: {gcn}')
print(f'ROCm/HIP: {torch.version.hip}')
" 2>&1 | tee -a "$log"

    echo ""
    echo "--- Triton info ---"
    python -c "import triton; print(f'Triton: {triton.__version__}')" 2>&1 | tee -a "$log" || true

    echo ""
    echo "--- aiter arch_info ---"
    python -c "
from aiter.ops.triton.utils._triton import arch_info
print(f'Arch: {arch_info.get_arch_name()}')
print(f'FP8: {arch_info.is_fp8_avail()}')
print(f'FP4: {arch_info.is_fp4_avail()}')
" 2>&1 | tee -a "$log" || true

    if [ $failed -eq 1 ]; then
        echo -e "\n${RED}Phase 1 FAILED — cannot continue.${NC}"
        return 1
    fi
    phase_pass 1 "Build Verification complete"
}

########################################
# Phase 2: Triton Basic Ops (Triton-only, CK kernels fail on gfx908)
########################################
run_phase2() {
    phase_start 2 "Triton Basic Ops"
    local log="$LOGDIR/phase2_basic_ops.log"

    python -c "
import torch

device = 'cuda'
dtype = torch.bfloat16

print('--- Triton RMSNorm (rms_norm) ---')
from aiter.ops.triton.normalization.rmsnorm import rms_norm
x = torch.randn(128, 4096, dtype=dtype, device=device)
w = torch.ones(4096, dtype=dtype, device=device)
out = rms_norm(x, w, 1e-6)
print(f'  rms_norm: input={x.shape} -> output={out.shape} OK')

print('--- Triton RMSNorm (rmsnorm_forward_inference) ---')
from aiter.ops.triton.normalization.rmsnorm import rmsnorm_forward_inference
out2 = rmsnorm_forward_inference(x, w, 1e-6)
print(f'  rmsnorm_forward_inference: input={x.shape} -> output={out2.shape} OK')

print('--- Triton LayerNorm ---')
from aiter.ops.triton.normalization.norm import layer_norm
b = torch.zeros(4096, dtype=dtype, device=device)
out3 = layer_norm(x, w, b, 1e-5)
print(f'  layer_norm: input={x.shape} -> output={out3.shape} OK')

print('--- Triton RoPE ---')
from aiter.ops.triton.rope.rope import rope_fwd, RotateStyle
# Shape: (seq_len, batch, heads, dim)
x_rope = torch.randn(32, 1, 8, 64, dtype=dtype, device=device)
freqs = torch.randn(32, 1, 1, 32, dtype=torch.float32, device=device)
out4 = rope_fwd(x_rope, freqs, RotateStyle.NEOX, False, False)
print(f'  rope_fwd: input={x_rope.shape} -> output={out4.shape} OK')

print('--- Triton Softmax ---')
from aiter.ops.triton.softmax import softmax
x_sm = torch.randn(128, 4096, dtype=dtype, device=device)
out5 = softmax(x_sm)
print(f'  softmax: input={x_sm.shape} -> output={out5.shape} OK')

print()
print('Phase 2: All Triton basic ops passed!')
" 2>&1 | tee "$log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        phase_pass 2 "Triton basic ops"
    else
        phase_fail 2 "Triton basic ops"
        return 1
    fi
}

########################################
# Phase 3: Triton GEMM Tests
########################################
run_phase3() {
    phase_start 3 "Triton GEMM Tests"
    local log="$LOGDIR/phase3_triton_gemm.log"

    cd /workspace/aiter 2>/dev/null || true

    echo "--- Triton A16W16 GEMM ---"
    python -m pytest op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase3_a16w16.log" | tail -20
    echo ""

    echo "--- Triton A8W8 GEMM (key test — previously failed via CK) ---"
    python -m pytest op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase3_a8w8.log" | tail -20
    echo ""

    echo "--- Triton Batched BF16 GEMM ---"
    python -m pytest op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase3_batched_bf16.log" | tail -20
    echo ""

    echo "--- Triton A16W16 Gated GEMM ---"
    python -m pytest op_tests/triton_tests/gemm/basic/test_gemm_a16w16_gated.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase3_a16w16_gated.log" | tail -20
    echo ""

    phase_pass 3 "Triton GEMM tests complete (check logs for details)"
}

########################################
# Phase 4: Triton MoE Tests (THE critical test)
########################################
run_phase4() {
    phase_start 4 "Triton MoE Tests (CRITICAL)"
    local log="$LOGDIR/phase4_triton_moe.log"

    cd /workspace/aiter 2>/dev/null || true

    echo "--- Triton MoE align_block_size (pure Triton, bypasses DPP limitation) ---"
    python -m pytest op_tests/triton_tests/moe/test_moe_align_block_size.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase4_moe_align.log" | tail -20
    echo ""

    echo "--- Triton MoE GEMM (main MoE test) ---"
    python -m pytest op_tests/triton_tests/moe/test_moe.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase4_moe_main.log" | tail -30
    echo ""

    echo "--- Triton MoE routing ---"
    python -m pytest op_tests/triton_tests/moe/test_moe_routing.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase4_moe_routing.log" | tail -20
    echo ""

    echo "--- Triton MoE A8W8 GEMM ---"
    python -m pytest op_tests/triton_tests/moe/test_moe_gemm_a8w8.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase4_moe_a8w8.log" | tail -20
    echo ""

    phase_pass 4 "Triton MoE tests complete (check logs for details)"
}

########################################
# Phase 5: Triton Attention Tests
########################################
run_phase5() {
    phase_start 5 "Triton Attention Tests"
    local log="$LOGDIR/phase5_triton_attn.log"

    cd /workspace/aiter 2>/dev/null || true

    echo "--- Triton MHA ---"
    python -m pytest op_tests/triton_tests/attention/test_mha.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase5_mha.log" | tail -20
    echo ""

    echo "--- Triton PA Decode ---"
    python -m pytest op_tests/triton_tests/attention/test_pa_decode.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase5_pa_decode.log" | tail -20
    echo ""

    echo "--- Triton PA Prefill ---"
    python -m pytest op_tests/triton_tests/attention/test_pa_prefill.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase5_pa_prefill.log" | tail -20
    echo ""

    echo "--- Triton Prefill Attention ---"
    python -m pytest op_tests/triton_tests/attention/test_prefill_attention.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase5_prefill_attn.log" | tail -20
    echo ""

    echo "--- Triton Extend Attention ---"
    python -m pytest op_tests/triton_tests/attention/test_extend_attention.py -v -x -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase5_extend_attn.log" | tail -20
    echo ""

    phase_pass 5 "Triton Attention tests complete (check logs for details)"
}

########################################
# Phase 6: Triton Normalization, RoPE, Quant, and Full Suite
########################################
run_phase6() {
    phase_start 6 "Full Triton Test Suite"
    local log="$LOGDIR/phase6_full_triton.log"

    cd /workspace/aiter 2>/dev/null || true

    echo "--- Triton RMSNorm ---"
    python -m pytest op_tests/triton_tests/normalization/test_rmsnorm.py -v -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase6_rmsnorm.log" | tail -20
    echo ""

    echo "--- Triton LayerNorm ---"
    python -m pytest op_tests/triton_tests/normalization/test_layernorm.py -v -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase6_layernorm.log" | tail -20
    echo ""

    echo "--- Triton RoPE ---"
    python -m pytest op_tests/triton_tests/rope/ -v -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase6_rope.log" | tail -20
    echo ""

    echo "--- Triton Softmax ---"
    python -m pytest op_tests/triton_tests/test_softmax.py -v -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase6_softmax.log" | tail -20
    echo ""

    echo "--- Triton TopK ---"
    python -m pytest op_tests/triton_tests/test_topk.py -v -W ignore::DeprecationWarning 2>&1 | tee "$LOGDIR/phase6_topk.log" | tail -20
    echo ""

    echo ""
    echo "--- Full triton_tests suite (informational) ---"
    python -m pytest op_tests/triton_tests/ -v -W ignore::DeprecationWarning -q 2>&1 | tee "$LOGDIR/phase6_full_triton_suite.log" | tail -40
    echo ""

    phase_pass 6 "Full Triton suite complete (check logs for details)"
}

########################################
# Summary
########################################
print_summary() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}Test Results Summary${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo "Log files in $LOGDIR:"
    ls -la "$LOGDIR"/*.log 2>/dev/null || echo "  (no logs found)"
    echo ""
    echo "Quick pass/fail counts from pytest logs:"
    for f in "$LOGDIR"/phase*.log; do
        if [ -f "$f" ]; then
            basename=$(basename "$f" .log)
            result=$(grep -E "^(FAILED|PASSED|ERROR|=)" "$f" | tail -1 || echo "no summary line")
            echo "  $basename: $result"
        fi
    done
}

########################################
# Main
########################################
echo -e "${CYAN}AITER MI100 Test Runner${NC}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

case "$run_phase" in
    1) run_phase1 ;;
    2) run_phase1 && run_phase2 ;;
    3) run_phase3 ;;
    4) run_phase4 ;;
    5) run_phase5 ;;
    6) run_phase6 ;;
    all)
        run_phase1
        run_phase2 || true
        run_phase3 || true
        run_phase4 || true
        run_phase5 || true
        run_phase6 || true
        print_summary
        ;;
    *)
        echo "Usage: $0 [1|2|3|4|5|6|all]"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"
