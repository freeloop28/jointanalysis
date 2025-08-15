import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W], raw logits
        # targets: [B, H, W], class indices
        inputs = torch.softmax(inputs, dim=1)  # Apply softmax for multi-class Dice

        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_onehot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

class TverskyLoss(nn.Module):
    """
    Tversky Loss for multi-class segmentation.

    Args:
        alpha (float): weight for false positives.
        beta (float): weight for false negatives.
        smooth (float): smoothing term to avoid division by zero.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: raw logits from model, shape (B, C, H, W)
            targets: ground truth labels, shape (B, H, W)
        """
        num_classes = inputs.shape[1]
        inputs = F.softmax(inputs, dim=1)  # apply softmax for multi-class probs

        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # dimensions to sum over: batch, height, width
        tp = torch.sum(inputs * targets_one_hot, dims)
        fp = torch.sum(inputs * (1 - targets_one_hot), dims)
        fn = torch.sum((1 - inputs) * targets_one_hot, dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1 - tversky
        return loss.mean()
