import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from datasets.bdd import BDD100KSegmentation
from models.segformer_b0 import create_segformer_b0
from utils.eval_utils import evaluate


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
if __name__ == "__main__":
    device = get_device()
    print("Using device: ", device)

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(PROJECT_ROOT, "data")

    # ---- Config ----
    image_size   = (256, 256)
    batch_size   = 1
    epochs       = 2
    lr           = 1e-4
    ignore_index = 255
    num_classes  = 20

    # ---- DataLoaders ----
    train_dataset = BDD100KSegmentation(root=data_root, split="train", image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("Train samples: ", len(train_dataset))
    val_dataset = BDD100KSegmentation(root=data_root, split="val", image_size=image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Val samples: ", len(val_dataset))

    # ---- Model, loss, optimizer ----
    model = create_segformer_b0(num_classes=num_classes, pretrained=True)
    model.to(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    # ---- Training loop ----
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)   # (B, 3, H, W)
            masks  = masks.to(device)    # (B, H, W)

            optimizer.zero_grad()

            outputs = model(images)
            logits = outputs["out"]     # (B, C, H, W)

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                avg_loss = running_loss / 20
                print(
                    f"[SegFormer] Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Train Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0
        # ---- Validation at end of epoch ----
        val_loss, val_acc = evaluate(model, val_loader, device, criterion, ignore_index=ignore_index)
        print(
            f"[SegFormer] End of epoch {epoch+1}: "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Pixel Acc = {val_acc:.4f}"
        )
        # --- Save best SegFormer model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_path, "segformer_b0_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best SegFormer model to {best_path}")
     # Save final weights
    last_path = os.path.join(checkpoint_path, "segformer_b0_last.pth")
    torch.save(model.state_dict(), last_path)
    print("SegFormer training finished. Final model saved to", last_path)



