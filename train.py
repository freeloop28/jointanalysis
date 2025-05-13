import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import LightUNet
from dataload import SpermSegmentationDataset
from utils.losses import DiceLoss, TverskyLoss
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from scipy.ndimage import label
from datetime import datetime
import uuid

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# --- Load YAML config ---
def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- Metric computation ---
def compute_metrics(preds, targets, num_classes):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    pixel_correct = (preds == targets).sum()
    pixel_total = preds.size
    pixel_acc = pixel_correct / pixel_total

    iou_list = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        iou = intersection / union if union != 0 else 0.0
        iou_list.append(iou)

    miou = np.mean(iou_list)
    return pixel_acc, miou, iou_list

# --- Shape-based tail filtering ---
def post_process_tail(tail_mask):
    tail_mask_np = tail_mask.cpu().numpy().astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(tail_mask_np, cv2.MORPH_OPEN, kernel, iterations=1)
    labeled, num = label(opened)
    filtered = np.zeros_like(opened)
    for region_label in range(1, num + 1):
        area = (labeled == region_label).sum()
        if area > 30:
            filtered[labeled == region_label] = 1
    return filtered

# --- Save visual results ---
def save_prediction_visuals(inputs, targets, preds, epoch, run_dir, class_colors, iou_list=None):
    save_dir = os.path.join(run_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(4, inputs.size(0))):
        img = TF.to_pil_image(inputs[i].cpu())
        gt = targets[i].cpu().numpy()
        pr = preds[i].cpu().numpy()

        gt_color = np.zeros((*gt.shape, 3), dtype=np.uint8)
        pr_color = np.zeros_like(gt_color)

        for cls, color in class_colors.items():
            gt_color[gt == cls] = color
            pr_color[pr == cls] = color

        overlay = np.array(img) * 0.4 + pr_color * 0.6
        overlay = overlay.astype(np.uint8)

        fig, axs = plt.subplots(1, 4, figsize=(12, 4))
        axs[0].imshow(img); axs[0].set_title("Input"); axs[0].axis('off')
        axs[1].imshow(gt_color); axs[1].set_title("GT Mask"); axs[1].axis('off')
        axs[2].imshow(pr_color); axs[2].set_title("Prediction"); axs[2].axis('off')
        axs[3].imshow(overlay); axs[3].set_title("Overlay"); axs[3].axis('off')

        if i == 0 and iou_list:
            for j, iou in enumerate(iou_list):
                axs[3].text(2, 15 + j * 20, f"Class {j} IoU: {iou:.3f}", color='white', fontsize=9, backgroundcolor='black')

        plt.tight_layout()
        unique_id = uuid.uuid4().hex[:6]
        filename = f"sample_{i}_epoch_{epoch}_{unique_id}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

# --- Save full training curves ---
def save_training_curves(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(df["epoch"], df["pixel_acc"], label="Pixel Accuracy")
    plt.title("Pixel Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(2, 2, 3)
    plt.plot(df["epoch"], df["mIoU"], label="Mean IoU")
    for i in range(3):
        plt.plot(df["epoch"], df[f"iou_cls_{i}"], label=f"IoU Class {i}")
    plt.title("IoU over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Train one epoch ---
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for inputs, targets, *_ in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# --- Validate one epoch ---
def validate_one_epoch(model, dataloader, loss_fn, device, num_classes, epoch, run_dir):
    model.eval()
    val_loss = 0.0
    total_preds, total_targets = [], []
    with torch.no_grad():
        for inputs, targets, *_ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()

            softmax_output = torch.softmax(outputs, dim=1)
            preds = torch.argmax(softmax_output, dim=1)

            for i in range(preds.size(0)):
                filtered_tail = post_process_tail(preds[i] == 1)
                preds[i][(preds[i] == 1) & (torch.tensor(filtered_tail, device=preds.device) == 0)] = 0

            total_preds.append(preds)
            total_targets.append(targets)

        all_preds = torch.cat(total_preds)
        all_targets = torch.cat(total_targets)
        pixel_acc, miou, iou_per_class = compute_metrics(all_preds, all_targets, num_classes)

        save_prediction_visuals(inputs, targets, preds, epoch, run_dir, class_colors={
            0: [0, 0, 0],
            1: [0, 255, 0],
            2: [255, 0, 0]
        }, iou_list=iou_per_class)

        return val_loss / len(dataloader), pixel_acc, miou, iou_per_class

# --- Main training loop ---
def main():
    config = load_config()
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    model_cfg = config['model']
    model = LightUNet(
        n_channels=model_cfg['in_channels'],
        n_classes=model_cfg['num_classes'],
        base_channels=model_cfg['base_channels'],
        bilinear=model_cfg['bilinear']
    ).to(device)

    weights = torch.tensor([0.02, 4.0, 1.0], device=device)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss()
    tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
    focal_loss = FocalLoss(alpha=weights, gamma=2)

    def loss_fn(inputs, targets):
        return 0.2 * ce_loss(inputs, targets) + 0.3 * focal_loss(inputs, targets) + \
               0.3 * tversky_loss(inputs, targets) + 0.2 * dice_loss(inputs, targets)

    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"📂 Loading data from: {config['data']['dataset_path']}")
    train_dataset = SpermSegmentationDataset(config['data']['dataset_path'], split='train')
    val_dataset = SpermSegmentationDataset(config['data']['dataset_path'], split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    log_records = []
    for epoch in range(1, config['train']['epochs'] + 1):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        avg_val_loss, pixel_acc, miou, iou_per_class = validate_one_epoch(
            model, val_loader, loss_fn, device, config['model']['num_classes'], epoch, run_dir
        )

        print(f"\n📘 Epoch {epoch}/{config['train']['epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f}")
        print(f"Pixel Accuracy: {pixel_acc:.4f}")
        print(f"mIoU: {miou:.4f}")
        print(f"IoU per class: {np.array2string(np.array(iou_per_class), precision=7)}")

        log_records.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'pixel_acc': pixel_acc,
            'mIoU': miou,
            **{f'iou_cls_{i}': iou for i, iou in enumerate(iou_per_class)}
        })

    df_log = pd.DataFrame(log_records)
    df_log.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)
    save_training_curves(df_log, os.path.join(run_dir, "training_curves.png"))
    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pth"))
    print("📊 Saved training log, curves, and final model.")

if __name__ == "__main__":
    main()