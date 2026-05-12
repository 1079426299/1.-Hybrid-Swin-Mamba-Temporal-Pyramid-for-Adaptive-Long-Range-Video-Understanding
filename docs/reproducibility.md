# Reproducibility guide

Record the following information when reporting experiments:

- Python version
- PyTorch version
- CUDA/cuDNN version
- GPU model and number of GPUs
- precision mode: FP32 or AMP/FP16
- batch size
- input frames and resolution
- clips/crops at inference

## Smoke test

```bash
python train.py --config configs/smtpn_tiny_dummy.yaml --epochs 1 --batch-size 2 --output outputs/smoke
python eval.py --config configs/smtpn_tiny_dummy.yaml --checkpoint outputs/smoke/checkpoint_last.pt
```

## Code availability statement template

Code, configuration files, evaluation scripts, and pretrained checkpoints required to reproduce the main results are publicly available at `[Zenodo DOI]`, with a development mirror hosted at `[GitHub URL]`.

## Data availability statement template

The benchmark datasets used in this study, including Kinetics-400, Something-Something V2, and EPIC-Kitchens-100, are publicly available from their respective providers under their own access terms. The processed metadata files and evaluation scripts generated in this study are available at `[Zenodo DOI]`.
