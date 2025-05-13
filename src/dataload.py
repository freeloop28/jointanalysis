import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class SpermSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', target_size=(120, 120), augment=True):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        self.label_map = {0: 0, 200: 1, 255: 2}  # Background, Tail, Head
        self.augment = augment

        # ⚙️ Define Albumentations transforms
        if split == 'train' and augment:
            self.transform = A.Compose([
                A.Resize(*self.target_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, interpolation=1, mask_interpolation=0, approximate=False, p=0.3),
                A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-15, 15), shear=(-5, 5), interpolation=1, mask_interpolation=0, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*self.target_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])

        all_videos = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        num_videos = len(all_videos)

        # 🌟 Split ratio
        num_train = int(0.7 * num_videos)
        num_val = int(0.15 * num_videos)
        video_splits = {
            'train': all_videos[:num_train],
            'val': all_videos[num_train:num_train + num_val],
            'test': all_videos[num_train + num_val:],
        }

        if split not in video_splits:
            raise ValueError(f"Unknown split: {split}")

        self.samples = []
        for video in video_splits[split]:
            video_path = os.path.join(root_dir, video)
            sperm_ids = sorted([
                d for d in os.listdir(video_path)
                if os.path.isdir(os.path.join(video_path, d))
            ])

            for sperm_id in sperm_ids:
                sperm_path = os.path.join(video_path, sperm_id)
                track_path = os.path.join(sperm_path, 'track.json')
                if not os.path.exists(track_path):
                    continue

                with open(track_path, 'r') as f:
                    track_data = json.load(f)

                for fname in sorted(os.listdir(sperm_path)):
                    if fname.endswith('_head_and_tail.png'):
                        image_name = fname.replace('_head_and_tail.png', '')
                        mask_path = os.path.join(sperm_path, fname)

                        image_path = os.path.join(sperm_path, image_name + '.jpg')
                        if not os.path.exists(image_path):
                            image_path = os.path.join(sperm_path, image_name + '.png')
                        if not os.path.exists(image_path):
                            print(f"⚠️ Image not found: {image_path}")
                            continue

                        self.samples.append({
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'track_data': track_data.get(image_name, None),
                            'image_name': image_name,
                        })

        print(f"[INFO] {split.upper()} set: {len(video_splits[split])} videos, {len(self.samples)} samples.")

        if len(self.samples) == 0:
            print(f"⚠️ Warning: No valid samples found in {split} set!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.array(Image.open(sample['image_path']).convert('RGB'))
        mask = np.array(Image.open(sample['mask_path']).convert('L'))

        label = np.zeros_like(mask, dtype=np.int64)
        for val, cls in self.label_map.items():
            label[mask == val] = cls

        augmented = self.transform(image=image, mask=label)
        image_tensor = augmented['image']
        label_tensor = augmented['mask'].long()

        if idx == 0 and self.split == 'train':
            print(f"[DEBUG] Image shape: {image_tensor.shape}, min: {image_tensor.min():.3f}, max: {image_tensor.max():.3f}")
            print(f"[DEBUG] Label unique values: {torch.unique(label_tensor)}")

        return image_tensor, label_tensor, sample['track_data'], sample['image_name']
