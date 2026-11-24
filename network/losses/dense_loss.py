import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSegmentationLoss(nn.Module):
    """Loss function for semantic segmentation using cross-entropy and Dice terms.

    Attributes:
        nc (int): Number of semantic classes.
        ignore_index (int): Label value of invalid pixels excluded from the loss.
        ce (nn.CrossEntropyLoss): Cross-entropy loss with configurable ignore_index.
    """

    def __init__(self, model: torch.nn.Module, alpha=0.3, beta=0.7, reduction='none', ignore_index=255):
        """Initialize semantic segmentation loss.

        Args:
            model (torch.nn.Module): Model containing the SemanticSegment head.
            ignore_index (int): Target value that will be ignored in loss computation.
        """
        super().__init__()
        m = model.model[-1]
        self.nc = m.nc
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        weight = getattr(model, "class_weights", None)  # cls_pw frequency weights, else hardcoded Cityscapes
        weight = None if weight is None else weight.to(device=self.device, dtype=self.dtype)
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss(reduction=reduction)  # binary: class weighting intentionally unsupported
        else:
            self.ce = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index).to(
                device=self.device, dtype=self.dtype
            )
            if weight is not None:
                # Non-persistent: weight is a deterministic constant, no need to serialize into ckpt state_dict.
                self.ce.register_buffer("weight", weight, persistent=False)

    def _resize_masks(self, masks, target_shape):
        """Resize masks to match prediction spatial dimensions."""
        if masks.shape[1:] != target_shape:
            return (
                F.interpolate(masks.float().unsqueeze(1), size=target_shape, mode="nearest").squeeze(1).to(torch.int32)
            )
        return masks

    def _ce_loss(self, preds, masks):
        """Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path."""
        if self.nc == 1:
            flat = masks.reshape(-1)
            valid = flat != self.ignore_index
            logits = preds.reshape(-1)[valid]
            target = flat[valid].float()
        else:
            logits = preds
            target = masks

        return self.ce(logits, target)

    def _dice_loss(self, preds, masks):
        """Compute Dice loss excluding ignore pixels."""
        if self.nc == 1:
            return self._binary_dice_loss(preds, masks)
        flat_target = masks.reshape(-1)
        valid = flat_target != self.ignore_index
        if not valid.any():
            return preds.sum() * 0

        pred_soft = F.softmax(preds, dim=1)
        target = flat_target[valid].long()
        flat_pred = pred_soft.float().permute(0, 2, 3, 1).reshape(-1, self.nc)[valid]
        intersection = torch.zeros(self.nc, device=preds.device, dtype=torch.float32)
        intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
        pred_sum = flat_pred.sum(dim=0)
        target_sum = torch.bincount(target, minlength=self.nc).to(device=preds.device, dtype=torch.float32)
        cardinality = pred_sum + target_sum
        return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

    def _binary_dice_loss(self, preds, masks):
        """Compute Dice loss for single-class (binary) segmentation.

        Pixels equal to ignore_index are excluded from Dice terms to match BCE valid-pixel filtering.
        """
        valid = (masks != self.ignore_index).float()
        pred_soft = preds.squeeze(1).sigmoid()
        target = (masks == 1).float()
        intersection = (pred_soft * target * valid).sum()
        cardinality = ((pred_soft + target) * valid).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

    def forward(self, preds, batch):
        """Compute semantic segmentation loss with optional auxiliary loss.

        Args:
            preds (torch.Tensor | tuple): Main logits [B, nc, H', W'], or (main, aux) tuple.
            batch (dict): Batch dict with 'semantic_mask' [B, H, W] containing class IDs
                (ignore_index marks invalid pixels).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): (total_loss * batch_size, detached loss items [ce, dice, aux]).
        """
        # Unpack auxiliary logits when present.
        aux_logits = None
        if isinstance(preds, tuple):
            preds, aux_logits = preds

        masks = batch["semantic_mask"].to(preds.device)
        if preds.shape[2:] != masks.shape[1:]:
            preds = F.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)

        # Main cross-entropy and Dice loss.
        ce_loss = self._reduce_ce(self._ce_loss(preds, masks), masks)
        dice_loss = self._dice_loss(preds, masks)
        total = (ce_loss + dice_loss) * self.alpha
        # Auxiliary cross-entropy loss. Match ce_loss dtype so torch.stack below succeeds under AMP.
        aux_loss = torch.tensor(0.0, device=preds.device, dtype=ce_loss.dtype)
        if aux_logits is not None:
            if aux_logits.shape[2:] != masks.shape[1:]:
                aux_logits = F.interpolate(aux_logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            aux_loss = self._reduce_ce(self._ce_loss(aux_logits, masks), masks) * self.beta
            total += aux_loss

        loss_items = torch.stack([ce_loss, dice_loss, aux_loss]).detach()
        return total * preds.shape[0], loss_items

    def _reduce_ce(self, ce_map, masks):
        """Average CE over valid pixels when reduction is none."""
        if not torch.is_tensor(ce_map) or ce_map.ndim == 0 or ce_map.numel() == 1:
            return ce_map
        # Multiclass CE keeps a spatial map; binary BCE already drops ignore pixels.
        if ce_map.shape == masks.shape:
            valid = masks != self.ignore_index
            if not valid.any():
                return ce_map.sum() * 0
            return ce_map[valid].mean()
        return ce_map.mean()


class DenseLoss(nn.Module):
    """Criterion for computing training losses for semantic segmentation task."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.2, gate=0.05, reduction='none', ignore_index=255):
        """Initialize DenseLoss.

        Args:
            ignore_index (int): Label of invalid pixels excluded from CE / Dice / gate loss.
            alpha (float): Weight of the first (aux) prediction branch.
            beta (float): Weight of the second (main) prediction branch.
            gamma (float): Weight of the hard-pixel gate loss.
            gate (float): Per-pixel CE threshold; only pixels above it contribute to gate loss.
            reduction (str): Reduction used by CrossEntropyLoss; keep 'none' to enable gate.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gate = gate
        self.use_gate = gate > 0
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def _ce_loss(self, preds, masks):
        """Compute cross-entropy, ignoring pixels equal to ignore_index."""
        if preds.shape[1] == 1:
            flat = masks.reshape(-1)
            valid = flat != self.ignore_index
            if not valid.any():
                return preds.sum() * 0
            logits = preds.reshape(-1)[valid]
            target = flat[valid].float()
            return F.binary_cross_entropy_with_logits(logits, target, reduction=self.reduction)
        return self.ce(preds, masks.long())

    def _reduce_ce(self, ce_map, masks):
        """Average CE over valid pixels when reduction is none."""
        if not torch.is_tensor(ce_map) or ce_map.ndim == 0 or ce_map.numel() == 1:
            return ce_map
        # Multiclass CE keeps a spatial map; binary BCE already drops ignore pixels.
        if ce_map.shape == masks.shape:
            valid = masks != self.ignore_index
            if not valid.any():
                return ce_map.sum() * 0
            return ce_map[valid].mean()
        return ce_map.mean()

    def _dice_loss(self, preds, masks):
        """Compute Dice loss excluding ignore pixels."""
        if preds.shape[1] == 1:
            return self._binary_dice_loss(preds, masks)
        nc = preds.shape[1]
        flat_target = masks.reshape(-1)
        valid = flat_target != self.ignore_index
        if not valid.any():
            return preds.sum() * 0

        pred_soft = F.softmax(preds, dim=1)
        target = flat_target[valid].long()
        flat_pred = pred_soft.float().permute(0, 2, 3, 1).reshape(-1, nc)[valid]
        intersection = torch.zeros(nc, device=preds.device, dtype=torch.float32)
        intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
        pred_sum = flat_pred.sum(dim=0)
        target_sum = torch.bincount(target, minlength=nc).to(device=preds.device, dtype=torch.float32)
        cardinality = pred_sum + target_sum
        return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

    def _binary_dice_loss(self, preds, masks):
        """Compute Dice loss for single-class (binary) segmentation."""
        valid = (masks != self.ignore_index).float()
        pred_soft = preds.squeeze(1).sigmoid()
        target = (masks == 1).float()
        intersection = (pred_soft * target * valid).sum()
        cardinality = ((pred_soft + target) * valid).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

    def _gate_loss(self, preds, gt_mask):
        """Hard-pixel CE averaged over valid pixels above the gate threshold."""
        loss_map = self._ce_loss(preds, gt_mask)
        if not torch.is_tensor(loss_map) or loss_map.ndim == 0 or loss_map.numel() == 1:
            return loss_map
        if loss_map.shape == gt_mask.shape:
            valid = gt_mask != self.ignore_index
            loss_gs = loss_map[valid & (loss_map > self.gate)]
        else:
            # Binary path already dropped ignore pixels.
            loss_gs = loss_map[loss_map > self.gate]
        if loss_gs.numel() == 0:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)
        return loss_gs.mean()

    def _branch_loss(self, preds, gt_mask):
        """CE + Dice for a single prediction branch."""
        ce = self._reduce_ce(self._ce_loss(preds, gt_mask), gt_mask)
        dice = self._dice_loss(preds, gt_mask)
        return ce + dice

    def forward(self, preds, gt_mask):
        """Calculate the loss for semantic segmentation."""
        gt_mask = gt_mask.long()

        if isinstance(preds, torch.Tensor):
            loss = self._branch_loss(preds, gt_mask)
            if self.use_gate:
                loss = loss + self.gamma * self._gate_loss(preds, gt_mask)
            return loss

        loss_0 = self._branch_loss(preds[0], gt_mask)
        loss_1 = self._branch_loss(preds[1], gt_mask)
        loss = self.alpha * loss_0 + self.beta * loss_1
        if self.use_gate:
            loss_g0 = self._gate_loss(preds[0], gt_mask)
            loss_g1 = self._gate_loss(preds[1], gt_mask)
            loss_g = self.alpha * loss_g0 + self.beta * loss_g1
            loss = loss + self.gamma * loss_g
        return loss
