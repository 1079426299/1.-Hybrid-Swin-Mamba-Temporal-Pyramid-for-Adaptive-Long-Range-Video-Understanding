import argparse
import json
from pathlib import Path
import torch
from datasets import build_dataloaders
from models import build_model
from engine import train_one_epoch, evaluate
from utils import load_config, set_seed, ensure_dir, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train SMTPN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/default")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
        cfg["evaluation"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
        cfg["evaluation"]["num_workers"] = args.num_workers

    ensure_dir(args.output)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device)
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"].get("lr", 1e-3)), weight_decay=float(cfg["training"].get("weight_decay", 0.05)))

    history = []
    for epoch in range(1, int(cfg["training"].get("epochs", 1)) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, cfg)
        val_metrics = evaluate(model, val_loader, device, cfg)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))
        save_checkpoint(str(Path(args.output) / "checkpoint_last.pt"), model, optimizer, epoch, cfg)

    with open(Path(args.output) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
