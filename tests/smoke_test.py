import torch
from models import build_model

def test_smtpn_forward():
    cfg = {
        "dataset": {"channels": 3},
        "model": {
            "num_classes": 10,
            "embed_dims": [16, 32, 64, 128],
            "mecam_state_dim": 16,
            "pyramid_levels": [1, 2, 3],
        },
    }
    model = build_model(cfg)
    x = torch.randn(2, 3, 8, 64, 64)
    out = model(x, return_aux=True)
    assert out["logits"].shape == (2, 10)
    assert len(out["aux_logits"]) == 3
