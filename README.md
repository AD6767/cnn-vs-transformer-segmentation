# CNN vs Transformer Segmentation (BDD-style)

This repo is a small comparison project between a **CNN-based** model and a future **Transformer-based** model for **semantic segmentation** on a driving dataset (BDD100K-style).

Right now, the repo contains a working **CNN baseline** using DeepLabV3-ResNet50. The Transformer part will be added later.

---

## Setup

Clone the repo and create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
```

---

## Data layout

The code expects a BDD-style layout under `data/`:

```text
data/
  images/
    seg/
      train/*.jpg
      val/*.jpg
  labels/
    seg/
      train/*.png   # class-id masks
      val/*.png
```
Each mask is a single-channel PNG with integer class IDs; 255 is treated as ignore/void.

---

## Usage

### Training (CNN baseline)
```bash
python train.py
```
* Uses datasets/BDD100KSegmentation to load `data/images/seg` and `data/labels/seg`
* Trains a DeepLabV3-ResNet50 model with per-pixel cross-entropy loss.
* Runs validation after each epoch (loss + pixel accuracy).
* Saves checkpoints to checkpoints/:
    * deeplab_best.pth - best validation loss
    * deeplab_last.pth - final model at the end of training.

### Evaluation & visualization
```bash
python eval.py
```
* Loads `checkpoints/deeplab_best.pth`.
* Evaluates on the val split (same metrics as in training).
* Saves a few example visualizations to viz_deeplab/
    * Input image
    * Ground-truth mask
    * Predicted mask
