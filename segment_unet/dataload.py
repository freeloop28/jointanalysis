# dataload.py — dataset + albumentations factory compatible with the new config structure

"""A re‑worked SpermSegmentationDataset that:

* **does not rely on a global `config` variable** — the caller passes a `data_cfg` dict.
* Falls back to sensible defaults if `data_cfg` is omitted, so existing scripts keep运行.
* Uses the config to switch individual augmentations on/off (hflip, vflip, rotate, affine, color jitter).
* Supports arbitrary `image_size`: int → square, or (h, w) tuple.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from skimage.morphology import skeletonize  # noqa: F401  (kept for future use)
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# Transform helpers
# -----------------------------------------------------------------------------

_DEF_AUG = {
    "hflip": True,
    "vflip": True,
    "rotate": True,
    "affine": True,
    "color_jitter": True,
}


def _ensure_size(size_cfg: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    """Convert `image_size` config to (H, W) tuple."""
    if isinstance(size_cfg, int):
        return (size_cfg, size_cfg)
    if isinstance(size_cfg, (list, tuple)) and len(size_cfg) == 2:
        return tuple(size_cfg)  # type: ignore[arg-type]
    raise ValueError("`image_size` must be int or (h,w) tuple/list")


def build_train_transforms(data_cfg: Dict[str, Any]) -> A.Compose:
    """Create an Albumentations Compose according to `data_cfg`."""
    h, w = _ensure_size(data_cfg["image_size"])
    aug_cfg = {**_DEF_AUG, **data_cfg.get("aug", {})}

    ops: List[A.BasicTransform] = [
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR)
    ]

    # Optional augmentations
    if aug_cfg.get("hflip", True):
        ops.append(A.HorizontalFlip(p=0.5))
    if aug_cfg.get("vflip", True):
        ops.append(A.VerticalFlip(p=0.3))
    if aug_cfg.get("rotate", True):
        ops.append(A.RandomRotate90(p=0.5))
    if aug_cfg.get("affine", True):
        ops.append(
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.05, 0.05),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.2,
            )
        )
    if aug_cfg.get("color_jitter", True):
        ops.append(
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.3)
        )

    # Final normalisation & tensor conversion
    ops += [
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    return A.Compose(ops)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class SpermSegmentationDataset(Dataset):
    """Sperm head/tail segmentation with optional data augmentation."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        target_size: Tuple[int, int] | None = None,
        augment: bool = True,
        data_cfg: Dict[str, Any] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        # target_size fallback if data_cfg missing
        self._target_size = (
            target_size if target_size is not None else (120, 120)
        )
        self.augment = augment and split == "train"

        # ------------- label map -------------
        # Background 0, Tail 1, Head 2  (original mask values: 0, 200, 255)
        self.label_map = {0: 0, 200: 1, 255: 2}

        # ------------- transforms -------------
        if self.augment:
            if data_cfg is None:
                # build minimal cfg from defaults
                data_cfg = {
                    "image_size": self._target_size,
                    "aug": _DEF_AUG,
                }
            self.transform = build_train_transforms(data_cfg)
        else:
            # validation / test — deterministic resize + normalise
            h, w = _ensure_size(
                data_cfg["image_size"] if data_cfg and "image_size" in data_cfg else self._target_size
            )
            self.transform = A.Compose(
                [
                    A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )

        # ------------- scan folder structure -------------
        all_videos = sorted([d for d in os.listdir(self.root_dir) if (self.root_dir / d).is_dir()])
        num_videos = len(all_videos)
        num_train = int(0.7 * num_videos)
        num_val = int(0.15 * num_videos)
        video_splits = {
            "train": all_videos[:num_train],
            "val": all_videos[num_train : num_train + num_val],
            "test": all_videos[num_train + num_val :],
        }
        if split not in video_splits:
            raise ValueError(f"Unknown split '{split}' — choose from train/val/test")

        self.samples: List[Dict[str, Any]] = []
        for vid in video_splits[split]:
            video_path = self.root_dir / vid
            for sperm_dir in sorted(video_path.iterdir()):
                if not sperm_dir.is_dir():
                    continue
                track_path = sperm_dir / "track.json"
                track_data = {}
                if track_path.exists():
                    with open(track_path, "r", encoding="utf-8") as f:
                        track_data = json.load(f)

                for png in sperm_dir.glob("*_head_and_tail.png"):
                    image_stem = png.stem.replace("_head_and_tail", "")
                    # prefer .jpg else .png
                    img_path = sperm_dir / f"{image_stem}.jpg"
                    if not img_path.exists():
                        img_path = sperm_dir / f"{image_stem}.png"
                    if not img_path.exists():
                        # skip if no image counterpart
                        continue

                    self.samples.append(
                        {
                            "image_path": img_path,
                            "mask_path": png,
                            "track_data": track_data.get(image_stem),
                            "image_name": image_stem,
                        }
                    )

        print(
            f"[INFO] {split.upper()} set — videos: {len(video_splits[split])}, samples: {len(self.samples)}"
        )
        if not self.samples:
            print(f"[WARN] No valid samples found in '{split}' split!")

    # ---------------------------------------------------------------------
    # Dunder methods
    # ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = np.array(Image.open(sample["image_path"]).convert("RGB"))
        mask_raw = np.array(Image.open(sample["mask_path"]).convert("L"))

        # remap labels
        mask = np.zeros_like(mask_raw, dtype=np.int64)
        for val, cls in self.label_map.items():
            mask[mask_raw == val] = cls

        # future: skeleton processing (kept for parity with old code)
        # tail_mask = (mask_raw == 200).astype(np.uint8)
        # if tail_mask.sum() > 0:
        #     mask[skeletonize(tail_mask) == 1] = 2

        augmented = self.transform(image=img, mask=mask)
        return (
            augmented["image"],  # Tensor CHW float32 [-1,1]
            augmented["mask"].long(),
            sample["track_data"],
            sample["image_name"],
        )

