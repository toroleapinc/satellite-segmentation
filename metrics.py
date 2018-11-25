"""Evaluation metrics for segmentation."""
import numpy as np

def iou_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.float32)
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)

def mean_iou(preds, targets, num_classes=10, threshold=0.5):
    ious = []
    for cls in range(num_classes):
        iou = iou_score(preds[:, cls], targets[:, cls], threshold)
        ious.append(iou)
    return np.mean(ious), ious

def dice_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.float32)
    inter = (pred_bin * target).sum()
    return (2 * inter) / (pred_bin.sum() + target.sum() + 1e-8)
