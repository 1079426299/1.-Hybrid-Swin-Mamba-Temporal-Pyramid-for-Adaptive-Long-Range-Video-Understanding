import random
from pathlib import Path
import yaml
import numpy as np
import torch


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))
    results = []
    for k in topk:
        k = min(k, logits.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / target.numel()).item())
    return results


def save_checkpoint(path: str, model, optimizer, epoch: int, cfg):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None, "epoch": epoch, "config": cfg}, path)


def load_checkpoint(path: str, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt
