import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import torch
from models import build_model
from utils import load_config, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize temporal offsets predicted by DTPN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="outputs/offsets.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
    dcfg = cfg["dataset"]
    x = torch.randn(1, int(dcfg.get("channels", 3)), int(dcfg["frames"]), int(dcfg["height"]), int(dcfg["width"]), device=device)
    outputs = model(x)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for idx, off in enumerate(outputs["offsets"]):
        plt.plot(off[0].detach().cpu().numpy(), label=f"level {idx}")
    plt.xlabel("Temporal index")
    plt.ylabel("Predicted temporal offset")
    plt.title("DTPN temporal offsets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved offset visualization to {args.output}")


if __name__ == "__main__":
    main()
