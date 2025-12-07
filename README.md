# CNN vs Transformer Segmentation (BDD-style)

Mini project to compare a **CNN-based** model and a **Transformer-based** model for **semantic segmentation** on a driving dataset (BDD100K-style):

- CNN: **DeepLabV3-ResNet50**
- Transformer: **SegFormer-B0** (via segmentation_models_pytorch)

Both use the same dataset, loss, training loop, and evaluation code.

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

### Training 

#### CNN baseline (DeepLabV3-ResNet50)
```bash
python train_deeplab_cnn.py
```
* Saves checkpoints to:
    * checkpoints/deeplab_best.pth - best validation loss
    * checkpoints/deeplab_last.pth - final model at the end of training.

#### Transformer baseline (SegFormer-B0)
```bash
python train_segformer.py
```
Saves checkpoints to:
* checkpoints/segformer_b0_best.pth
* checkpoints/segformer_b0_last.pth

### Evaluation & visualization
```bash
# CNN baseline
python eval.py --model deeplab

# Transformer baseline
python eval.py --model segformer
```
* Load the corresponding `*_best.pth` checkpoint.
* Evaluates on the val split (same metrics as in training).
* Save visualizations to:
  * viz_deeplab/ (input | GT | DeepLab prediction)
  * viz_segformer/ (input | GT | SegFormer prediction)

---

## Current results (256×256, BDD-style val split)
On the same validation set:
```
Model        Val Loss    Val Pixel Acc
--------------------------------------
DeepLab      0.4197      0.8723
SegFormer    0.4452      0.8692
```
With this shallow training setup, the CNN baseline is slightly ahead in both loss and pixel accuracy. Visualizations in `viz_deeplab/` and `viz_segformer/` make it easy to compare failure modes (eg: object boundaries, small/far objects, thin structures) between the two architectures.

---

## Future work
Some possible extensions:
1. Train longer / tune per-model: More epochs, learning-rate schedules, and model-specific hyperparams (eg: weight decay, warmup) for a fairer comparison.
2. Better metrics: Add per-class IoU / mIoU and confusion matrices, not just pixel accuracy.
3. More models: Try other backbones -- larger SegFormer variants (B1/B2..), or other ViT-style decoders.
4. Compare impact of input resolution, data augmentation, and class subsets (eg: only “drivable area + vehicles”).

---

## Notes: CNN vs Transformer in segmentation
#### When might transformers beat CNNs?
Transformers like SegFormer tend to shine when:
* You have more data and longer training (they're more data-hungry).
* The task needs strong global context, eg: understanding far-away objects and relationships.
* You use strong pretraining (large-scale ImageNet / multi-dataset pretraining) and then fine-tune.
* You move toward more open-world / complex settings (panoptic, multi-task, multi-camera).
In small-data, low-epoch regimes with simple training, a good CNN baseline can be competitive or even slightly better, as seen here.

#### Pros / cons
**CNNs (like DeepLabV3-ResNet50)**  
- 👍 Good at capturing **local details** (edges, textures, small shapes).  
- 👍 Usually **easier to train** and stable with less data and fewer epochs.  
- 👍 Often **faster and lighter**, good as a strong baseline.  
- 👎 Don't naturally see the **whole image at once**; global context is limited.

**Transformers (like SegFormer-B0)**  
- 👍 Very good at **global context**: how different parts of the image relate.  
- 👍 Can scale well with **more data and longer training**.  
- 👎 Often **need more data / training** to clearly beat a good CNN.  
- 👎 Can be **heavier** and more sensitive to training setup.
