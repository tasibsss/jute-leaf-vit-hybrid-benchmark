# Jute Leaf Disease Classification: ViT, Swin-Tiny, EfficientNetB0 and a Lightweight-Attention Hybrid

Code, split file, and per-run predictions for the paper "Benchmarking Vision Transformers, CNNs, and a Lightweight Attention-Augmented Hybrid for Jute Leaf Disease Classification: An Interpretability-Focused Comparative Study" (anonymized for review).

## Dataset
Public jute leaf disease collection (Islam & Sheikh, *Data in Brief*, 2025, doi:10.1016/j.dib.2025.112334): 1,390 original images, five classes (Fresh, Dieback, Holed, Mosaic, Stem Soft Rot). The images are **not** included here; download them from the data descriptor and use the 1,390 original images, not the pre-augmented version.

## Contents
- `<notebook>.ipynb`: full pipeline (training, evaluation, statistics, cost, ablation, Grad-CAM, localization, duplicate check). Runs end to end.
- `dataset_split.csv`: fixed stratified 70/20/10 split (973 train / 278 val / 139 test), reused by every model.
- `predictions/`: per-run test predictions (4 models x 5 seeds).
- `annotations/`: LabelMe rectangle boxes for the 23 annotated test images (single annotator).

## Setup
Kaggle notebook, NVIDIA Tesla T4, PyTorch + HuggingFace Transformers, `pytorch-grad-cam`, `timm`/torchvision. Install the packages the first cell requests.

## Models
- Plain ViT: `google/vit-base-patch16-224-in21k`
- Hybrid: ViT + lightweight token attention (reduction r=4, single head, residual + LayerNorm + dropout 0.1)
- Swin-Tiny, EfficientNetB0 (ImageNet-pretrained)

## Training configuration
AdamW (lr 3e-5, weight decay 0.01), batch size 64, up to 100 epochs, class-weighted cross-entropy with label smoothing 0.1, fp16 mixed precision. Early stopping on validation loss (patience 10, min delta 5e-4); the best-validation-accuracy checkpoint is evaluated on the test set. Seeds: 42, 123, 2024, 7, 2025 (same for all models). The seeds change initialization, augmentation, and batch order, not the split.

## Evaluation
Accuracy and macro-averaged precision/recall/F1 (mean +/- sample std over five seeds), 95% Wilson intervals, exact McNemar tests with Holm correction, parameters, GFLOPs (2 x MACs), and median batch-1 fp32 latency. Grad-CAM analysis: deletion/insertion AUC, top-10% heat share and entropy (all 139 test images), and heat-in-box, pointing-game hit rate, and lift (23 annotated images).

## Reproducibility note
GPU kernels are not fully deterministic: re-executing the notebook changed 2 of 20 runs by one test image each. All reported numbers come from one complete execution (about 104 minutes for the 20 main runs).

## License
Code released for review purposes. The dataset is subject to its original license.
