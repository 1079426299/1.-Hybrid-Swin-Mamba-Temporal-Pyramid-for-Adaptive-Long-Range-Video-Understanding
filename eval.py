import argparse
import json
import torch
from datasets import build_dataloaders
from models import build_model
from engine import evaluate
from utils import load_config, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SMTPN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["evaluation"]["batch_size"] = args.batch_size
    device = torch.device(args.device)
    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
    print(json.dumps(evaluate(model, val_loader, device, cfg), indent=2))


if __name__ == "__main__":
    main()
