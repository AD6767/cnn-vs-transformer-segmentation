import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.bdd import BDD100KSegmentation
from models.deeplab_resnet50 import create_deeplab_resnet50
from models.segformer_b0 import create_segformer_b0
from utils.eval_utils import evaluate, visualize_batch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument(
        "--model",
        choices=["deeplab", "segformer"],
        default="deeplab",
        help="Which model to evaluate: 'deeplab' or 'segformer'",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pth). If not set, use default for the model.",
    )
    parser.add_argument(
        "--num-vis-batches",
        type=int,
        default=3,
        help="How many batches to visualize (each batch has batch_size images).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    print("Using device:", device)
    print("Evaluating model:", args.model)

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(PROJECT_ROOT, "data")

    # ---- Config (consistent with training) ----
    image_size   = (256, 256)
    batch_size   = 2
    num_classes  = 20
    ignore_index = 255
    num_vis_batches = 3

    # ---- Val dataset & loader ----
    val_dataset = BDD100KSegmentation(
        root=data_root,
        split="val",
        image_size=image_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    print("Val samples:", len(val_dataset))

    # ---- Select model + default checkpoint ----
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    if args.model == "deeplab":
        model = create_deeplab_resnet50(num_classes=num_classes, pretrained=True)
        default_ckpt = os.path.join(checkpoint_dir, "deeplab_best.pth")
    else:  # "segformer"
        model = create_segformer_b0(num_classes=num_classes, pretrained=True)
        default_ckpt = os.path.join(checkpoint_dir, "segformer_b0_best.pth")

    checkpoint_path = args.checkpoint or default_ckpt
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Train first (train.py for deeplab, train_segformer.py for segformer)."
        )
    # ---- Load weights ----
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ---- Criterion (same as training) ----
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # ---- Evaluate on full val set ----
    val_loss, val_acc = evaluate(model, val_loader, device, criterion, ignore_index)
    print(f"[Eval] Val Loss = {val_loss:.4f}, Val Pixel Acc = {val_acc:.4f}")

    # ---- Visualize a few batches ----
    viz_dir = os.path.join(PROJECT_ROOT, f"viz_{args.model}")
    os.makedirs(viz_dir, exist_ok=True)

    print(f"Saving visualizations to {viz_dir} ...")
    with torch.no_grad():
        start_idx = 0
        for b_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            logits  = outputs["out"]
            preds   = logits.argmax(dim=1)  # (B, H, W)

            visualize_batch(
                images=images,
                masks=masks,
                preds=preds,
                out_dir=viz_dir,
                start_index=start_idx,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            start_idx += images.size(0)

            if b_idx + 1 >= num_vis_batches:
                break

    print("Done. Check the PNGs in:", viz_dir)
