import numpy as np
import torch
import torch.nn as nn


def ohem_single(score, gt_text, training_mask):
    pos_num = int(np.sum(gt_text > 0.5)) - int(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float64')
        return selected_mask

    neg_num = int(np.sum(gt_text <= 0.5))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float64')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float64')
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    """
    Input shape: B x H x W
    Output type: bool
    """
    scores = scores.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, axis=0)
    selected_masks = torch.from_numpy(selected_masks).bool().cuda()  # Recommended, but not compatible with torch==1.1.0
    # selected_masks = torch.from_numpy(selected_masks).to(torch.uint8).cuda()  # For compatibility with torch==1.1.0
    return selected_masks


class Criterion:

    def __init__(self, class_weights):
        self.class_weights = class_weights

        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def masked_ce_with_logits(self, pred, target, mask):
        """Input shape: B x H x W"""
        return self.bce_with_logits(pred[mask], target[mask]) if torch.any(mask) else 0.

    def masked_smooth_l1(self, pred, target, mask):
        """Input shape: B x H x W"""
        return self.smooth_l1(pred[mask], target[mask]) if torch.any(mask) else 0.

    def __call__(self, pred, target, ignored_mask):
        mask_ohnm = ohem_batch(pred[:, 0], target[:, 0], 1. - ignored_mask)  # bool.
        mask_others = target[:, 0] == 1.

        l_cls_1 = self.masked_ce_with_logits(pred[:, 0], target[:, 0], mask_ohnm)
        l_cls_2 = self.masked_ce_with_logits(pred[:, 1], target[:, 1], mask_others)
        l_cls_3 = self.masked_ce_with_logits(pred[:, 2], target[:, 2], mask_others)
        l_reg_1 = self.masked_smooth_l1(pred[:, 3], target[:, 3], mask_others)
        l_reg_2 = self.masked_smooth_l1(pred[:, 4], target[:, 4], mask_others)
        l_reg_3 = self.masked_smooth_l1(pred[:, 5], target[:, 5], mask_others)

        total_loss = 0.
        for idx, loss in enumerate((l_cls_1, l_cls_2, l_cls_3, l_reg_1, l_reg_2, l_reg_3)):
            total_loss += self.class_weights[idx] * loss
        return total_loss / sum(self.class_weights)
