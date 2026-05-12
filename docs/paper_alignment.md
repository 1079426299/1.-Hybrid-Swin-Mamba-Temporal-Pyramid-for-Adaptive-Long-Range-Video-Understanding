# Alignment between manuscript components and repository files

| Manuscript component | Repository implementation |
|---|---|
| Hierarchical Swin backbone | `models/backbone.py` |
| ME-CAM selective state-space memory | `models/me_cam.py` |
| Deformable temporal pyramid | `models/deformable_temporal_pyramid.py` |
| SMTPN full model | `models/smtpn.py` |
| Cross-scale semantic consistency loss | `models/smtpn.py::semantic_consistency_loss` |
| Training objective | `engine.py` |
| Runtime and memory profiling | `tools/profile_runtime.py` |
| Temporal offset visualization | `tools/visualize_offsets.py` |
| Temporal variance computation | `tools/compute_temporal_variance.py` |

## Implementation note

The manuscript describes deformable temporal alignment using DCNv2-style adaptive sampling. This repository uses a pure PyTorch temporal sampler for portability. If a CUDA DCNv2 implementation is used for final experiments, report the exact package version and keep the same interface.
