import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from models import build_model
from utils import load_config, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Export SMTPN to ONNX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="outputs/smtpn.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = build_model(cfg).eval()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location="cpu")
    dcfg = cfg["dataset"]
    x = torch.randn(1, int(dcfg.get("channels", 3)), int(dcfg["frames"]), int(dcfg["height"]), int(dcfg["width"]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, x, args.output, opset_version=args.opset, input_names=["video"], output_names=["outputs"], dynamic_axes={"video": {0: "batch", 2: "frames"}})
    print(f"Saved ONNX model to {args.output}")


if __name__ == "__main__":
    main()
