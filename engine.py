import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.smtpn import semantic_consistency_loss
from utils import accuracy


def train_one_epoch(model, loader, optimizer, device, epoch: int, cfg):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"].get("amp", False)) and device.type == "cuda")
    consistency_weight = float(cfg["model"].get("consistency_weight", 0.0))
    clip_grad_norm = cfg["training"].get("clip_grad_norm", None)
    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for videos, targets in pbar:
        videos = videos.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(videos, return_aux=True)
            cls_loss = F.cross_entropy(outputs["logits"], targets)
            con_loss = semantic_consistency_loss(outputs["logits"], outputs.get("aux_logits", []))
            loss = cls_loss + consistency_weight * con_loss
        scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        acc1 = accuracy(outputs["logits"].detach(), targets, topk=(1,))[0]
        running_loss += loss.item()
        running_acc += acc1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc1=f"{acc1:.2f}")
    n = max(len(loader), 1)
    return {"loss": running_loss / n, "acc1": running_acc / n}


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    pbar = tqdm(loader, desc="Eval", leave=False)
    for videos, targets in pbar:
        videos = videos.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(videos, return_aux=False)
        loss = F.cross_entropy(outputs["logits"], targets)
        acc1, acc5 = accuracy(outputs["logits"], targets, topk=(1, 5))
        running_loss += loss.item()
        running_acc1 += acc1
        running_acc5 += acc5
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc1=f"{acc1:.2f}", acc5=f"{acc5:.2f}")
    n = max(len(loader), 1)
    return {"loss": running_loss / n, "acc1": running_acc1 / n, "acc5": running_acc5 / n}
