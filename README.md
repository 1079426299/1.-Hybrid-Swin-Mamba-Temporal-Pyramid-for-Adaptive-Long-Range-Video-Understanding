# SMTPN: Swin-Mamba Temporal Pyramid Network

This repository provides a runnable reference implementation for **SMTPN**, a hybrid temporal pyramid framework for adaptive long-range video understanding.

Implemented manuscript components:

- hierarchical video feature extraction;
- **ME-CAM**: Mamba-Enhanced Contextual Attention Memory for selective state-space temporal modeling;
- **DTPN**: Deformable Temporal Pyramid Network for adaptive multi-scale temporal alignment;
- cross-scale semantic consistency loss;
- training, evaluation, profiling, and visualization utilities.

> This repository is associated with the manuscript submitted to *The Visual Computer*. Please cite the manuscript and this repository if you use the code.

## Quick start

```bash
pip install -r requirements.txt
python train.py --config configs/smtpn_tiny_dummy.yaml --epochs 1 --batch-size 2 --output outputs/smoke
python eval.py --config configs/smtpn_tiny_dummy.yaml --checkpoint outputs/smoke/checkpoint_last.pt --batch-size 2
python tools/profile_runtime.py --config configs/smtpn_tiny_dummy.yaml --batch-size 1 --frames 16 --height 112 --width 112
```

The default quick-start config uses synthetic video data and does not require any public benchmark download.

## Dataset CSV format

For real datasets, prepare CSV files with two columns:

```text
video_path,label
/path/to/video_0001.mp4,12
/path/to/video_0002.mp4,31
```

Then update one of the dataset configs in `configs/`.

## Repository structure

```text
SMTPN/
├── configs/
├── data/
├── datasets/
├── docs/
├── models/
├── scripts/
├── tests/
├── tools/
├── train.py
├── eval.py
├── engine.py
├── requirements.txt
├── environment.yml
├── LICENSE
└── CITATION.cff
```

## Implementation note

The manuscript describes deformable temporal alignment using DCNv2-style adaptive sampling. To keep this repository portable and runnable without custom CUDA kernels, `models/deformable_temporal_pyramid.py` implements temporal deformable sampling with PyTorch `grid_sample`. An optimized DCNv2/CUDA implementation can be plugged in later while preserving the same API.

## Citation

```bibtex
@misc{smtpn2026,
  title={Hybrid Swin-Mamba Temporal Pyramid Network for Adaptive Long-Range Video Understanding},
  author={Your Name and Collaborators},
  year={2026},
  note={Code and models available at the repository DOI}
}
```
