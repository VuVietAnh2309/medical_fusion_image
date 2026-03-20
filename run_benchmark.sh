#!/usr/bin/env bash
# =============================================================================
# Benchmark 4 models × 2 datasets (PET-MRI, SPECT-MRI)
# Chạy từ root: /data2/anhvv/ai_nlp_rs/
# Usage: bash run_benchmark.sh [inference|eval|all]
# =============================================================================

set -e
ROOT=/data2/anhvv/ai_nlp_rs
PYTHON=$(which python)   # dùng python của env đang active (mri)
DATASETS="PET-MRI SPECT-MRI"

# ─────────────────────────────────────────────────────────────
# STEP 1: INFERENCE
# ─────────────────────────────────────────────────────────────

run_inference() {

# ── CDDFuse ──────────────────────────────────────────────────
# Checkpoint: CDDFuse_MIF.pth (Medical Image Fusion)
# Datasets:   MRI_PET (42 imgs), MRI_SPECT (73 imgs) — đã có sẵn trong test_img/
# Output:     CDDFuse/test_result/MRI_PET/, CDDFuse/test_result/MRI_SPECT/
echo "=== [1/4] CDDFuse inference ==="
cd $ROOT/CDDFuse
$CONDA_RUN python test_MIF.py
cd $ROOT

# ── MambaDFuse ───────────────────────────────────────────────
# Checkpoint: Model/Medical_Fusion-PET-MRI/    (PET-MRI)
#             Model/Medical_Fusion-SPECT-MRI/  (SPECT-MRI)
# Output:     MambaDFuse/results/MambaDFuse_PET-MRI/    (25 imgs — đã có)
#             MambaDFuse/results/MambaDFuse_SPECT-MRI/

echo "=== [2/4] MambaDFuse inference ==="
cd $ROOT/MambaDFuse

# PET-MRI (skip nếu đã có đủ 24 ảnh)
PET_COUNT=$(ls results/MambaDFuse_PET-MRI/*.png 2>/dev/null | wc -l)
if [ "$PET_COUNT" -ge 24 ]; then
    echo "  PET-MRI: already done ($PET_COUNT images), skipping"
else
    $CONDA_RUN python test_MambaDFuse.py \
        --model_path ./Model/Medical_Fusion-PET-MRI/Medical_Fusion/models/ \
        --iter_number 10000 \
        --root_path $ROOT/datasets \
        --dataset data_PET-MRI/test \
        --A_dir PET \
        --B_dir MRI \
        --in_channel 1
fi

# SPECT-MRI
$CONDA_RUN python test_MambaDFuse.py \
    --model_path ./Model/Medical_Fusion-SPECT-MRI/Medical_Fusion/models/ \
    --iter_number 10000 \
    --root_path $ROOT/datasets \
    --dataset data_SPECT-MRI/test \
    --A_dir SPECT \
    --B_dir MRI \
    --in_channel 1

cd $ROOT

# ── SwinFusion ───────────────────────────────────────────────
# Checkpoint: Model/Medical_Fusion-PET-MRI/   (PET-MRI ✅)
#             SPECT-MRI: không có checkpoint → BỎ QUA
# Output:     SwinFusion/results/SwinFusion_PET-MRI/

echo "=== [3/4] SwinFusion inference ==="
cd $ROOT/SwinFusion

# PET-MRI
$CONDA_RUN python test_swinfusion.py \
    --model_path ./Model/Medical_Fusion-PET-MRI/Medical_Fusion/models/ \
    --iter_number 10000 \
    --root_path $ROOT/datasets \
    --dataset data_PET-MRI/test \
    --A_dir PET \
    --B_dir MRI \
    --in_channel 1

# SPECT-MRI → SKIP (no checkpoint)
echo "  SPECT-MRI: SKIP — no checkpoint available"

cd $ROOT

# ── IFCNN ────────────────────────────────────────────────────
# Checkpoint: IFCNN-MAX.pth (general medical)
# Output:     IFCNN/Code/results/PET-MRI/, IFCNN/Code/results/SPECT-MRI/

echo "=== [4/4] IFCNN inference ==="
cd $ROOT/IFCNN/Code

$CONDA_RUN python infer_medical.py --dataset PET-MRI
$CONDA_RUN python infer_medical.py --dataset SPECT-MRI

cd $ROOT

echo ""
echo "=== Inference done ==="

} # end run_inference


# ─────────────────────────────────────────────────────────────
# STEP 2: EVAL 16 METRICS
# ─────────────────────────────────────────────────────────────

run_eval() {

echo ""
echo "============================================================"
echo "  BENCHMARK — 16 metrics (BASIC + LIU2012)"
echo "============================================================"

for DATASET in $DATASETS; do
    echo ""
    echo "############################################################"
    echo "  Dataset: $DATASET"
    echo "############################################################"

    for MODEL in CDDFuse MambaDFuse SwinFusion IFCNN; do
        $CONDA_RUN python eval.py --model $MODEL --dataset $DATASET --group benchmark 2>&1 \
            || echo "  [SKIP] $MODEL / $DATASET"
    done
done

echo ""
echo "=== Eval done ==="

} # end run_eval


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

MODE=${1:-all}

case $MODE in
    inference) run_inference ;;
    eval)      run_eval ;;
    all)       run_inference && run_eval ;;
    *)
        echo "Usage: bash run_benchmark.sh [inference|eval|all]"
        exit 1
        ;;
esac
