from pathlib import Path
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_video
from torchvision.transforms.functional import resize


class DummyVideoDataset(Dataset):
    """Synthetic video dataset for smoke testing."""

    def __init__(self, num_samples: int, num_classes: int, frames: int, height: int, width: int, channels: int = 3):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.frames = frames
        self.height = height
        self.width = width
        self.channels = channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        video = torch.randn(self.channels, self.frames, self.height, self.width, generator=g)
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        return video, label


class CsvVideoDataset(Dataset):
    """CSV video dataset. CSV columns: video_path,label"""

    def __init__(self, csv_file: str, frames: int, height: int, width: int, channels: int = 3):
        self.csv_file = Path(csv_file)
        self.frames = frames
        self.height = height
        self.width = width
        self.channels = channels
        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_file}. Use configs/smtpn_tiny_dummy.yaml for a synthetic smoke test."
            )
        self.samples = []
        with self.csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "video_path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV must contain columns: video_path,label")
            for row in reader:
                self.samples.append((row["video_path"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video: torch.Tensor) -> torch.Tensor:
        if video.numel() == 0:
            raise RuntimeError("Decoded an empty video.")
        video = video.float() / 255.0 if video.dtype == torch.uint8 else video.float()
        t = video.shape[0]
        if t >= self.frames:
            indices = torch.linspace(0, t - 1, self.frames).long()
            video = video[indices]
        else:
            pad = video[-1:].repeat(self.frames - t, 1, 1, 1)
            video = torch.cat([video, pad], dim=0)
        video = video.permute(0, 3, 1, 2)
        video = resize(video, [self.height, self.width], antialias=True)
        return video.permute(1, 0, 2, 3).contiguous()

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video, _, _ = read_video(path, pts_unit="sec")
        return self._sample_frames(video), torch.tensor(label, dtype=torch.long)


def build_dataloaders(cfg):
    dcfg = cfg["dataset"]
    batch_size = int(cfg.get("training", {}).get("batch_size", 2))
    eval_batch_size = int(cfg.get("evaluation", {}).get("batch_size", batch_size))
    num_workers = int(cfg.get("training", {}).get("num_workers", 0))
    eval_workers = int(cfg.get("evaluation", {}).get("num_workers", num_workers))

    if dcfg["name"] == "dummy":
        dataset = DummyVideoDataset(
            num_samples=int(dcfg.get("num_samples", 32)),
            num_classes=int(dcfg["num_classes"]),
            frames=int(dcfg["frames"]),
            height=int(dcfg["height"]),
            width=int(dcfg["width"]),
            channels=int(dcfg.get("channels", 3)),
        )
        val_len = max(1, len(dataset) // 4)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    elif dcfg["name"] == "csv":
        train_set = CsvVideoDataset(dcfg["train_csv"], dcfg["frames"], dcfg["height"], dcfg["width"], dcfg.get("channels", 3))
        val_set = CsvVideoDataset(dcfg["val_csv"], dcfg["frames"], dcfg["height"], dcfg["width"], dcfg.get("channels", 3))
    else:
        raise ValueError(f"Unknown dataset name: {dcfg['name']}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, num_workers=eval_workers, pin_memory=True)
    return train_loader, val_loader
