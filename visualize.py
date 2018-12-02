"""Overlay predictions on satellite images."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = plt.cm.tab10(np.arange(10))[:, :3]

def overlay_masks(image, masks, alpha=0.4, save=None):
    """Overlay class masks on image."""
    img = image.copy().astype(np.float32)
    if img.max() > 1:
        img = img / img.max()
    overlay = img.copy()
    for cls in range(masks.shape[-1]):
        mask = masks[:, :, cls] > 0.5
        for ch in range(3):
            overlay[:, :, ch][mask] = alpha * COLORS[cls][ch] + (1 - alpha) * img[:, :, ch][mask]
    if save:
        plt.figure(figsize=(12, 12))
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(save, bbox_inches='tight', dpi=150)
        plt.close()
    return overlay
