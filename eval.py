"""
Central Evaluation Script — Image Fusion Benchmark
====================================================
Chạy từ root: /data2/anhvv/ai_nlp_rs/

Usage:
    python eval.py --model MambaDFuse --dataset CT-MRI
    python eval.py --model FusionMamba --dataset PET-MRI
    python eval.py --model CDDFuse --dataset SPECT-MRI
    python eval.py --model SwinFusion --dataset CT-MRI
    python eval.py --model IFCNN --dataset PET-MRI

    # Chạy tất cả model trên một dataset
    python eval.py --dataset CT-MRI --all

    # Chỉ định thư mục fused thủ công
    python eval.py --fused_dir ./path/to/fused --dataset CT-MRI

    # Chọn nhóm metrics
    python eval.py --model MambaDFuse --dataset CT-MRI --group benchmark   # 16 metrics (mặc định)
    python eval.py --model MambaDFuse --dataset CT-MRI --group all         # 26 metrics
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.metrics import (
    evaluate_all,
    BENCHMARK_METRICS, FUSIONMAMBA_METRICS, LIU2012_METRICS, BASIC_METRICS, ALL_METRICS
)

# ──────────────────────────────────────────────────────────────
# Config: fused output directory của từng model × dataset
# ──────────────────────────────────────────────────────────────
MODELS = {
    'FusionMamba': {
        'CT-MRI':    'models/FusionMamba/outputs/CT-MRI',
        'PET-MRI':   'models/FusionMamba/outputs/PET-MRI',
        'SPECT-MRI': 'models/FusionMamba/outputs/SPECT-MRI',
    },
    'MambaDFuse': {
        'CT-MRI':    'models/MambaDFuse/results/MambaDFuse_data_CT-MRI/test',
        'PET-MRI':   'models/MambaDFuse/results/MambaDFuse_PET-MRI',
        'SPECT-MRI': 'models/MambaDFuse/results/MambaDFuse_data_SPECT-MRI/test',
    },
    'CDDFuse_IVF': {
        'CT-MRI':    'models/CDDFuse/test_result/MRI_CT_CDDFuse_IVF',
        'PET-MRI':   'models/CDDFuse/test_result/MRI_PET_CDDFuse_IVF',
        'SPECT-MRI': 'models/CDDFuse/test_result/MRI_SPECT_CDDFuse_IVF',
    },
    'CDDFuse_MIF': {
        'CT-MRI':    'models/CDDFuse/test_result/MRI_CT_CDDFuse_MIF',
        'PET-MRI':   'models/CDDFuse/test_result/MRI_PET_CDDFuse_MIF',
        'SPECT-MRI': 'models/CDDFuse/test_result/MRI_SPECT_CDDFuse_MIF',
    },
    'SwinFusion': {
        'CT-MRI':    'models/SwinFusion/results/SwinFusion_data_CT-MRI/test',
        'PET-MRI':   'models/SwinFusion/results/SwinFusion_data_PET-MRI/test',
        'SPECT-MRI': 'models/SwinFusion/results/SwinFusion_data_SPECT-MRI/test',
    },
    'IFCNN': {
        'CT-MRI':    'models/IFCNN/Code/results/CT-MRI',
        'PET-MRI':   'models/IFCNN/Code/results/PET-MRI',
        'SPECT-MRI': 'models/IFCNN/Code/results/SPECT-MRI',
    },
    'NestFuse': {
        'CT-MRI':    'models/NestFuse/results/CT-MRI',
        'PET-MRI':   'models/NestFuse/results/PET-MRI',
        'SPECT-MRI': 'models/NestFuse/results/SPECT-MRI',
    },
    'MUFusion': {
        'CT-MRI':    'models/MUFusion/medical/results/CT-MRI',
        'PET-MRI':   'models/MUFusion/medical/results/PET-MRI',
        'SPECT-MRI': 'models/MUFusion/medical/results/SPECT-MRI',
    },
    'LRFNet': {
        'CT-MRI':    'models/LRFNet/results/CT-MRI',
        'PET-MRI':   'models/LRFNet/results/PET-MRI',
        'SPECT-MRI': 'models/LRFNet/results/SPECT-MRI',
    },
    'EMMA': {
        'CT-MRI':    'models/EMMA/results/CT-MRI',
        'PET-MRI':   'models/EMMA/results/PET-MRI',
        'SPECT-MRI': 'models/EMMA/results/SPECT-MRI',
    },
    'DDFM': {
        'CT-MRI':    'models/DDFM/results/CT-MRI',
        'PET-MRI':   'models/DDFM/results/PET-MRI',
        'SPECT-MRI': 'models/DDFM/results/SPECT-MRI',
    },
    'DIDFuse': {
        'CT-MRI':    'models/DIDFuse/results/CT-MRI',
        'PET-MRI':   'models/DIDFuse/results/PET-MRI',
        'SPECT-MRI': 'models/DIDFuse/results/SPECT-MRI',
    },
}

# ──────────────────────────────────────────────────────────────
# Source dirs riêng cho từng model (override DATASETS nếu có)
# CDDFuse dùng test_img nội bộ nên source phải khớp với đó
# ──────────────────────────────────────────────────────────────
_cddfuse_sources = {
    'CT-MRI':    ('models/CDDFuse/test_img/MRI_CT/CT',      'models/CDDFuse/test_img/MRI_CT/MRI'),
    'PET-MRI':   ('models/CDDFuse/test_img/MRI_PET/PET',    'models/CDDFuse/test_img/MRI_PET/MRI'),
    'SPECT-MRI': ('models/CDDFuse/test_img/MRI_SPECT/SPECT','models/CDDFuse/test_img/MRI_SPECT/MRI'),
}
MODEL_SOURCES = {
    'CDDFuse_IVF': _cddfuse_sources,
    'CDDFuse_MIF': _cddfuse_sources,
}

# ──────────────────────────────────────────────────────────────
# Source dirs theo dataset
# ──────────────────────────────────────────────────────────────
DATASETS = {
    'CT-MRI': {
        'source1': 'datasets/data_CT-MRI/test/CT',
        'source2': 'datasets/data_CT-MRI/test/MRI',
    },
    'PET-MRI': {
        'source1': 'datasets/data_PET-MRI/test/PET',
        'source2': 'datasets/data_PET-MRI/test/MRI',
    },
    'SPECT-MRI': {
        'source1': 'datasets/data_SPECT-MRI/test/SPECT',
        'source2': 'datasets/data_SPECT-MRI/test/MRI',
    },
}


def resolve(path):
    """Convert relative path to absolute from ROOT."""
    return os.path.join(ROOT, path) if not os.path.isabs(path) else path


def run_eval(model_name, dataset, fused_dir, metrics, color):
    # Dùng source riêng của model nếu có, fallback về DATASETS chung
    if model_name in MODEL_SOURCES and dataset in MODEL_SOURCES[model_name]:
        s1, s2 = MODEL_SOURCES[model_name][dataset]
        source1_dir = resolve(s1)
        source2_dir = resolve(s2)
    else:
        ds = DATASETS[dataset]
        source1_dir = resolve(ds['source1'])
        source2_dir = resolve(ds['source2'])
    fused_dir   = resolve(fused_dir)

    print(f"\n{'═' * 65}")
    print(f"  Model:   {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Fused:   {fused_dir}")
    print(f"  Src1:    {source1_dir}")
    print(f"  Src2:    {source2_dir}")
    print(f"  Metrics: {len(metrics)} | Color: {color}")
    print(f"{'═' * 65}")

    if not os.path.isdir(fused_dir):
        print(f"  [SKIP] Fused dir not found: {fused_dir}")
        return None

    return evaluate_all(source1_dir, source2_dir, fused_dir, metrics, color)


def main():
    parser = argparse.ArgumentParser(description='Image Fusion Benchmark Evaluation')
    parser.add_argument('--model',   type=str, choices=list(MODELS.keys()),
                        help='Tên model cần eval')
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()), required=True,
                        help='Dataset: CT-MRI | PET-MRI | SPECT-MRI')
    parser.add_argument('--fused_dir', type=str, default=None,
                        help='Override thư mục fused output (không cần --model)')
    parser.add_argument('--all',    action='store_true',
                        help='Chạy tất cả models trên dataset chỉ định')
    parser.add_argument('--group',  type=str, default='benchmark',
                        choices=['benchmark', 'fusionmamba', 'liu2012', 'basic', 'all'],
                        help='benchmark=16 (mặc định) | fusionmamba=4 | liu2012=12 | all=22')
    parser.add_argument('--color',  type=str, default='gray',
                        choices=['gray', 'ycbcr'],
                        help='gray (mặc định cho medical) | ycbcr')
    args = parser.parse_args()

    # Chọn metric group
    group_map = {
        'benchmark':   BENCHMARK_METRICS,
        'fusionmamba': FUSIONMAMBA_METRICS,
        'liu2012':     LIU2012_METRICS,
        'basic':       BASIC_METRICS,
        'all':         ALL_METRICS,
    }
    metrics = group_map[args.group]

    if args.fused_dir:
        # Chạy thủ công với fused_dir
        model_name = args.model or 'Custom'
        run_eval(model_name, args.dataset, args.fused_dir, metrics, args.color)
        return

    if args.all:
        # Chạy tất cả models
        models_to_run = list(MODELS.keys())
    elif args.model:
        models_to_run = [args.model]
    else:
        parser.error('Cần chỉ định --model hoặc --all')

    for model_name in models_to_run:
        fused_rel = MODELS[model_name].get(args.dataset)
        if not fused_rel:
            print(f"\n[SKIP] {model_name} không có config cho {args.dataset}")
            continue
        run_eval(model_name, args.dataset, fused_rel, metrics, args.color)


if __name__ == '__main__':
    main()
