"""train.py — U-Net segmentation with Hydra + Weights & Biases
Logs loss / mIoU / pixel-accuracy and extra val metrics (precision/recall/F1),
plus per-class Precision-Recall curves and Average Precision (AP).
Saves model_best.pth whenever val/mIoU improves, and model_final.pth at the end.
"""

# ============================
# Imports & utilities
# ============================
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, average_precision_score  # PR curves

from dataload import SpermSegmentationDataset
from model import UNet
from utils.losses import DiceLoss, TverskyLoss, FocalLoss

# -----------------------------
# Config loading
# -----------------------------

def load_config(path: str = "config_loss_combo.yaml") -> dict:
    """Read YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -----------------------------
# Metrics helpers
# -----------------------------

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Compute pixel accuracy, mIoU *and* Precision/Recall/F1 (per-class and macro)."""
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    pixel_acc = (preds_np == targets_np).sum() / preds_np.size

    iou_list, prec_list, rec_list, f1_list = [], [], [], []
    for cls in range(num_classes):
        pred_cls = preds_np == cls
        target_cls = targets_np == cls

        tp = np.logical_and(pred_cls, target_cls).sum()
        fp = np.logical_and(pred_cls, np.logical_not(target_cls)).sum()
        fn = np.logical_and(np.logical_not(pred_cls), target_cls).sum()

        union = tp + fp + fn
        iou = tp / union if union else 0.0
        iou_list.append(iou)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        prec_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)

    metrics = {
        "pixel_acc": pixel_acc,
        "miou": float(np.mean(iou_list)),
        "iou_per_class": iou_list,
        "precision_per_class": prec_list,
        "recall_per_class": rec_list,
        "f1_per_class": f1_list,
        "precision_macro": float(np.mean(prec_list)),
        "recall_macro": float(np.mean(rec_list)),
        "f1_macro": float(np.mean(f1_list)),
    }
    return metrics

# -----------------------------
# Visualisation helpers
# -----------------------------

def save_prediction_visuals(inputs, targets, preds, epoch, run_dir, class_colors, iou_list=None):
    """Save overlay PNGs for the first 4 samples of the batch."""
    save_dir = Path(run_dir) / f"epoch_{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(4, inputs.size(0))):
        img_np = inputs[i].cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
        img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)

        gt = targets[i].cpu().numpy()
        pr = preds[i].cpu().numpy()

        gt_color = np.zeros((*gt.shape, 3), dtype=np.uint8)
        pr_color = np.zeros_like(gt_color)
        for cls, color in class_colors.items():
            gt_color[gt == cls] = color
            pr_color[pr == cls] = color

        overlay = (img_np * 0.4 + pr_color * 0.6).astype(np.uint8)

        fig, axs = plt.subplots(1, 4, figsize=(12, 4))
        for ax, im, title in zip(
            axs,
            [img_np, gt_color, pr_color, overlay],
            ["Input", "GT", "Prediction", "Overlay"],
        ):
            ax.imshow(im)
            ax.set_title(title)
            ax.axis("off")

        if i == 0 and iou_list:
            for j, iou in enumerate(iou_list):
                axs[3].text(
                    2,
                    15 + j * 18,
                    f"Class {j} IoU: {iou:.3f}",
                    color="white",
                    fontsize=8,
                    backgroundcolor="black",
                )
        plt.tight_layout()
        fname = save_dir / f"sample_{i}_epoch_{epoch}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(fname)
        plt.close()

        if wandb.run is not None:
            wandb.log({"val/overlay": wandb.Image(overlay, caption=f"epoch{epoch}_idx{i}")}, step=epoch)

# -----------------------------
# Training / validation loops
# -----------------------------

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for inputs, targets, *_ in tqdm(dataloader, desc="Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate_one_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    num_classes,
    epoch,
    run_dir,
    plot_pr_curves: bool = True,
    pr_sample_rate: float = 0.05,
):
    model.eval()
    val_loss = 0.0
    preds_all, targets_all = [], []

    viz_inputs = viz_targets = viz_preds = None
    y_true_chunks, y_prob_chunks = [], []

    with torch.no_grad():
        for batch_idx, (inputs, targets, *_) in enumerate(tqdm(dataloader, desc="Val", leave=False)):
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)  # (B,C,H,W)
            preds = torch.argmax(probs, dim=1)

            preds_all.append(preds)
            targets_all.append(targets)

            if batch_idx == 0:
                viz_inputs, viz_targets, viz_preds = inputs.detach().cpu(), targets.detach().cpu(), preds.detach().cpu()

            if plot_pr_curves:
                B, C, H, W = probs.shape
                prob_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # (N_pix, C)
                tgt_flat = targets.view(-1)                            # (N_pix,)
                if pr_sample_rate < 1.0:
                    N = tgt_flat.numel()
                    keep = torch.rand(N, device=tgt_flat.device) < pr_sample_rate
                    prob_flat = prob_flat[keep]
                    tgt_flat = tgt_flat[keep]
                y_true_chunks.append(tgt_flat.cpu().numpy())
                y_prob_chunks.append(prob_flat.cpu().numpy())

        preds_cat = torch.cat(preds_all)
        targets_cat = torch.cat(targets_all)

        metrics = compute_metrics(preds_cat, targets_cat, num_classes)

        wandb.log(
            {
                "val/loss": val_loss / len(dataloader),
                "val/pixel_acc": metrics["pixel_acc"],
                "val/miou": metrics["miou"],
                "val/precision": metrics["precision_macro"],
                "val/recall": metrics["recall_macro"],
                "val/f1": metrics["f1_macro"],
                **{f"val/iou_cls_{i}": v for i, v in enumerate(metrics["iou_per_class"])},
                **{f"val/prec_cls_{i}": v for i, v in enumerate(metrics["precision_per_class"])},
                **{f"val/rec_cls_{i}": v for i, v in enumerate(metrics["recall_per_class"])},
                **{f"val/f1_cls_{i}": v for i, v in enumerate(metrics["f1_per_class"])},
            },
            step=epoch,
        )

        if viz_inputs is not None:
            save_prediction_visuals(
                viz_inputs,
                viz_targets,
                viz_preds,
                epoch,
                run_dir,
                class_colors={0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]},
                iou_list=metrics["iou_per_class"],
            )

        if plot_pr_curves and len(y_true_chunks) > 0:
            y_true = np.concatenate(y_true_chunks, axis=0)  # (M,)
            y_prob = np.concatenate(y_prob_chunks, axis=0)  # (M, C)

            fig, ax = plt.subplots(figsize=(5, 4))
            ap_per_class = []
            for cls in range(num_classes):
                y_bin = (y_true == cls).astype(np.uint8)
                if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                    ap = 0.0
                    prec, rec = np.array([1.0]), np.array([0.0])
                else:
                    prec, rec, _ = precision_recall_curve(y_bin, y_prob[:, cls])
                    ap = average_precision_score(y_bin, y_prob[:, cls])
                ap_per_class.append(float(ap))
                ax.plot(rec, prec, label=f"cls{cls} AP={ap:.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Validation PR Curves (one-vs-rest)")
            ax.grid(True, linestyle="--", linewidth=0.5)
            ax.legend(fontsize=8)
            plt.tight_layout()

            if wandb.run is not None:
                wandb.log({"val/pr_curve": wandb.Image(fig)}, step=epoch)
                wandb.log({f"val/AP_cls_{i}": v for i, v in enumerate(ap_per_class)}, step=epoch)
            plt.close(fig)

        return val_loss / len(dataloader), metrics

# -----------------------------
# Main
# -----------------------------

def main():
    cfg = load_config()

    # device & RNG
    device = torch.device(
        cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(cfg["train"].get("seed", 42))

    # model
    mcfg = cfg["model"]
    model = UNet(
        n_channels=mcfg["in_channels"],
        n_classes=mcfg["num_classes"],
        #base_channels=mcfg["base_channels"],
        bilinear=mcfg.get("bilinear", True),
        use_attention=mcfg.get("use_attention", True),
    ).to(device)

    # loss
    loss_type = cfg["train"].get("loss_type", "combo")
    weights = torch.tensor([0.02, 1.0, 4.0], device=device)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss()
    tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
    focal_loss = FocalLoss(alpha=weights, gamma=2)

    if loss_type == "combo":
        def loss_fn(inp, tgt):
            return (
                0.2 * ce_loss(inp, tgt)
                + 0.3 * focal_loss(inp, tgt)
                + 0.3 * tversky_loss(inp, tgt)
                + 0.2 * dice_loss(inp, tgt)
            )
    elif loss_type == "dice_only":
        loss_fn = dice_loss
    elif loss_type == "CE_only":
        loss_fn = ce_loss
    else:
        raise ValueError(f"Unknown loss_type {loss_type}")

    # optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # data
    dcfg = cfg["data"]
    train_ds = SpermSegmentationDataset(dcfg["dataset_path"], split="train")
    val_ds = SpermSegmentationDataset(dcfg["dataset_path"], split="val")
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False
    )

    # output dirs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # W&B init
    wb_cfg = {
        **cfg,
        "run_dir": str(run_dir),
    }
    wandb.init(
        project=cfg.get("project", "unet_ablation"),
        name=cfg.get("name", f"run_{timestamp}"),
        group=cfg.get("group", "baseline"),
        config=wb_cfg,
    )

    # best-checkpoint tracking (maximize val/miou)
    best_miou = float("-inf")
    best_path = run_dir / "model_best.pth"

    # training loop
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            cfg["model"]["num_classes"],
            epoch,
            run_dir,
            plot_pr_curves=cfg["train"].get("plot_pr_curves", True),
            pr_sample_rate=cfg["train"].get("pr_sample_rate", 0.05),
        )

        wandb.log({"train/loss": train_loss}, step=epoch)

        print(
            f"\nEpoch {epoch}/{cfg['train']['epochs']}\n"
            f"Train Loss      : {train_loss:.4f}\n"
            f"Val Loss        : {val_loss:.4f}\n"
            f"Pixel Acc       : {val_metrics['pixel_acc']:.4f}\n"
            f"mIoU            : {val_metrics['miou']:.4f}\n"
            f"Macro Precision : {val_metrics['precision_macro']:.4f}\n"
            f"Macro Recall    : {val_metrics['recall_macro']:.4f}\n"
            f"Macro F1        : {val_metrics['f1_macro']:.4f}"
        )

        # save best by highest val/miou
        current = val_metrics["miou"]
        is_better = current > best_miou  # 若你想按最小保存，改成 "<"
        if is_better:
            best_miou = current
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "criterion": "max_val_miou",
                },
                best_path,
            )
            wandb.run.summary["best_miou"] = best_miou
            if wandb.run is not None:
                art_best = wandb.Artifact("unet_model_best", type="model")
                art_best.add_file(str(best_path))
                wandb.run.log_artifact(art_best, aliases=["best"])

    # save final
    model_path = run_dir / "model_final.pth"
    torch.save(model.state_dict(), model_path)
    art = wandb.Artifact("unet_model", type="model")
    art.add_file(str(model_path))
    wandb.run.log_artifact(art)
    wandb.finish()


if __name__ == "__main__":
    main()
