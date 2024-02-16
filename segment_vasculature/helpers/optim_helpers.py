import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate


class CEDiceFocalLoss(torch.nn.modules.loss._Loss):
    def __init__(self, ce_weight=0.9, dice_weight=0.01, focal_weight=0.09):
        super().__init__()
        self.ce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.focal = smp.losses.FocalLoss(mode="binary")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, preds, gt):
        return (
            self.ce_weight * self.ce(preds, gt)
            + self.dice_weight * self.dice(preds, gt)
            + self.focal_weight * self.focal(preds, gt)
        )


class BoundDiceFocalLoss(torch.nn.modules.loss._Loss):
    def __init__(self, bound_alpha=1.0, bound_weight=0.9, dice_weight=0.01, focal_weight=0.09):
        super().__init__()
        self.bound = EdgeEmphasisLoss(alpha=bound_alpha)
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.focal = smp.losses.FocalLoss(mode="binary")
        self.bound_weight = bound_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, preds, gt, boundaries):
        return (
            self.bound_weight * self.bound(preds, gt, boundaries)
            + self.dice_weight * self.dice(preds, gt)
            + self.focal_weight * self.focal(preds, gt)
        )


class BoundTwerskyFocalLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        bound_alpha=1.0,
        bound_weight=0.9,
        twersky_weight=0.01,
        twersky_alpha=0.5,
        twersky_beta=1,
        focal_weight=0.09,
    ):
        super().__init__()
        self.bound = EdgeEmphasisLoss(alpha=bound_alpha)
        self.twersky = smp.losses.TverskyLoss(mode="binary", alpha=twersky_alpha, beta=twersky_beta)
        self.focal = smp.losses.FocalLoss(mode="binary")
        self.bound_weight = bound_weight
        self.twersky_weight = twersky_weight
        self.focal_weight = focal_weight

    def forward(self, preds, gt, boundaries):
        return (
            self.bound_weight * self.bound(preds, gt, boundaries)
            + self.twersky_weight * self.twersky(preds, gt)
            + self.focal_weight * self.focal(preds, gt)
        )


class EdgeEmphasisLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(EdgeEmphasisLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets, boundaries):
        # Calculate standard binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Apply the edge weighting
        weighted_loss = bce_loss * (1 + self.alpha * boundaries)

        # Average over the batch
        return weighted_loss.mean()


def build_optim(cfg, model):
    optimizer, scheduler, criterion = None, None, None
    if hasattr(cfg, "optimizer"):
        optimizer = instantiate(cfg.optimizer, model.parameters())
    if hasattr(cfg, "scheduler"):
        scheduler = instantiate(cfg.scheduler.scheduler, optimizer)
    if hasattr(cfg, "loss"):
        criterion = instantiate(cfg.loss)  # , pos_weight=torch.ones([1], device='cuda')*2)

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
