import argparse
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from models import build_model
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Profile SMTPN runtime and memory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    frames = args.frames or int(dcfg["frames"])
    height = args.height or int(dcfg["height"])
    width = args.width or int(dcfg["width"])
    channels = int(dcfg.get("channels", 3))
    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()
    x = torch.randn(args.batch_size, channels, frames, height, width, device=device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.iters):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else None
    print({
        "params_m": round(params, 3),
        "batch_size": args.batch_size,
        "frames": frames,
        "height": height,
        "width": width,
        "throughput_videos_per_sec": round(args.batch_size * args.iters / elapsed, 3),
        "peak_memory_gb": None if peak_mem is None else round(peak_mem, 3),
        "device": str(device),
    })


if __name__ == "__main__":
    main()
