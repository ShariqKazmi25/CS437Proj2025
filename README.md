# Autonomous Trash Skimmer Boat · Vision Pipeline

An object-detection project for identifying floating debris in water-channel images. This repository contains the vision experiments, trained YOLOv8s checkpoints, and evaluation artifacts for the skimmer concept.

## Try the trained detector

In a Python environment with a compatible PyTorch build, install Ultralytics and run prediction on a local image or folder of images:

```bash
python -m pip install ultralytics==8.3.124
yolo detect predict model=best.pt source=path/to/images imgsz=320 device=cpu
```

Ultralytics saves annotated predictions under `runs/detect/`. The repository includes `best.pt` and `last.pt` at the root, along with [sample validation predictions](results/val_batch0_pred.jpg).

## Dataset and training

The experiments use the **Trash in Water Channels** dataset. See the [dataset project and annotations](https://github.com/abdulwahabamin/keras-yolo3-attention) and the [object-detection image archive](https://drive.google.com/file/d/1nQBjcg6-sU3yOxMzi0ybTwpQx4H5xuV0/view). The [dataset YAML](garbage.yaml) defines 12 classes, including plastic bags, bottles, and wrappers.

The [final notebook](Notebooks/26100101deliverable5.ipynb) shows how the annotations were converted to YOLO format and how the model was trained and evaluated. It uses Kaggle paths that need to be adjusted for a local run. Once images and labels are arranged under `images/train`, `images/val`, `labels/train`, and `labels/val`, point `garbage.yaml` at the dataset root and train with Ultralytics:

```bash
yolo detect train model=yolov8s.pt data=garbage.yaml epochs=20 imgsz=320 batch=4
```

See the [recorded training configuration](results/args.yaml) for the full run settings. Earlier notebooks cover a Keras YOLOv3 experiment and an earlier YOLOv8 run. The standalone `train.py` and `inference.py` files are exploratory drafts with incomplete dependencies and model wiring. Use the notebook or Ultralytics commands above to follow the documented experiments.

## Validation results

The [YOLOv8s training log](results/results.csv) records these validation metrics:

| Epoch | Precision | Recall | mAP@50 | mAP@50–95 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 76.1% | 24.8% | 25.2% | 14.7% |
| 20 | 78.9% | 56.1% | 61.8% | 39.4% |

The final notebook also tested CLIP ViT-B/32 as a post-processing step. It compared each detection crop with text prompts for the 12 classes, then combined the best similarity score with YOLO confidence. In that notebook's COCO bounding-box evaluation, mAP@50–95 was **31.9% for YOLOv8s** and **3.0% with CLIP rescoring**. The notebook applies a fused-score cutoff to the CLIP predictions, so these figures describe the tested configuration. The training-log metrics above come from Ultralytics validation.
