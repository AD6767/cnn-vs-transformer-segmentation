import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def evaluate(model, loader, device, criterion: nn.Module, ignore_index: int = 255):
    """
    Run model on a DataLoader and compute:
      - average loss over the dataset
      - pixel accuracy (optionally ignoring ignore_index)
    """
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)      # (B, H, W)

            outputs = model(images)
            logits = outputs["out"]       # (B, C, H, W)

            loss = criterion(logits, masks)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)  # (B, H, W)

            if ignore_index is not None:
                valid = masks != ignore_index
                correct = (preds == masks) & valid
                correct_pixels += correct.sum().item()
                total_pixels += valid.sum().item()
            else:
                correct_pixels += (preds == masks).sum().item()
                total_pixels += masks.numel()

    avg_loss = total_loss / len(loader.dataset)
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

    return avg_loss, pixel_acc


def unnormalize_image(tensor):
    """
    tensor: (3, H, W), normalized with ImageNet mean/std
    returns: (H, W, 3) in [0, 1] as numpy
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img = tensor.cpu().numpy()           # (3, H, W)
    img = img * std + mean               # de-normalize
    img = np.clip(img, 0.0, 1.0)
    img = np.transpose(img, (1, 2, 0))   # (H, W, 3)
    return img


def visualize_batch(images, masks, preds, out_dir, start_index=0,
                    num_classes=20, ignore_index=255):
    """
    Save side-by-side visualization: input | GT mask | predicted mask
    images: (B, 3, H, W)
    masks:  (B, H, W)
    preds:  (B, H, W)
    """
    os.makedirs(out_dir, exist_ok=True)
    cmap = plt.get_cmap("tab20")

    batch_size = images.size(0)
    for i in range(batch_size):
        img = unnormalize_image(images[i])
        gt  = masks[i].cpu().numpy()
        pr  = preds[i].cpu().numpy()

        if ignore_index is not None:
            gt_vis = gt.copy()
            gt_vis[gt_vis == ignore_index] = -1
        else:
            gt_vis = gt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img)
        axes[0].set_title("Input image")
        axes[0].axis("off")

        axes[1].imshow(gt_vis, vmin=0, vmax=num_classes - 1, cmap=cmap)
        axes[1].set_title("GT mask")
        axes[1].axis("off")

        axes[2].imshow(pr, vmin=0, vmax=num_classes - 1, cmap=cmap)
        axes[2].set_title("Pred mask")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"sample_{start_index + i:04d}.png")
        plt.savefig(out_path)
        plt.close(fig)
