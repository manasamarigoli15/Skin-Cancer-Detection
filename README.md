# Skin Cancer Detection Using Deep Learning and Explainable AI

End-to-end PyTorch project for classifying dermoscopic skin lesion images from the HAM10000 dataset. The project includes preprocessing, augmentation, EfficientNet and Vision Transformer training, model evaluation, Grad-CAM visual explanations, and a Streamlit inference app.

This software is for research and education only. It is not a medical device and must not be used as a substitute for professional clinical diagnosis.

## Demo Video 
```https://drive.google.com/file/d/1AS3wpuwDXT_5_Hgdofxd23CEbWYO0Bqz/view?usp=drive_link```

## Dataset

Downloaded HAM10000 from Kaggle 
HAM10000 labels used by this project:

| Label | Meaning |
| --- | --- |
| akiec | Actinic keratoses and intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch installation selector before installing the rest of the requirements.

## Configuration

Main settings live in `configs/config.yaml`:

- image size, batch size, workers, split ratios
- EfficientNet and ViT model names
- learning rate, epochs, weight decay, early stopping
- checkpoint, report, and app defaults

The default baseline is `efficientnet_b0`, and the comparison transformer is `vit_b_16` through `timm`.

## Train

Train EfficientNet:

```bash
python -m src.training.train --model efficientnet
```

Train ViT:

```bash
python -m src.training.train --model vit
```

Best checkpoints are saved to:

```text
checkpoints/best_efficientnet.pt
checkpoints/best_vit.pt
```

Training includes resizing, ImageNet normalization, random flips, random rotation, color jitter, weighted sampling, optional weighted loss, mixed precision on CUDA, logging, early stopping, and checkpointing.

## Evaluate

Evaluate EfficientNet:

```bash
python -m src.evaluation.evaluate --model efficientnet --checkpoint checkpoints/best_efficientnet.pt
```

Evaluate ViT:

```bash
python -m src.evaluation.evaluate --model vit --checkpoint checkpoints/best_vit.pt
```

Outputs include:

- `reports/<model>_metrics.json`
- `reports/<model>_classification_report.csv`
- `reports/figures/<model>_confusion_matrix.png`

Metrics reported are accuracy, macro precision, macro recall, macro F1-score, macro one-vs-rest ROC-AUC, and confusion matrix.

## Single-Image Inference

```bash
python -m src.models.infer \
  --model efficientnet \
  --checkpoint checkpoints/best_efficientnet.pt \
  --image path/to/lesion.jpg \
  --output-heatmap reports/figures/example_gradcam.png
```

The command prints the predicted class, confidence, probabilities, and a short explanation.

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app lets users upload a skin lesion image, choose a model checkpoint, view predicted class and confidence, inspect a Grad-CAM heatmap, and read a concise explanation of the model prediction.

## Notes on Grad-CAM

EfficientNet Grad-CAM uses the final convolutional feature layer. ViT visualization uses the final transformer block normalization output and reshapes patch tokens into a spatial grid. This provides a useful class-evidence map, but transformer explanations should be interpreted cautiously.

## Clean Coding Practices Included

- modular package structure
- YAML configuration
- reproducible seeding
- logging to `logs/`
- reusable dataset/model/evaluation utilities
- checkpoint save/load helpers
- CLI entry points for training, evaluation, and inference
- research-safety disclaimer for medical use

## Suggested Workflow

1. Download HAM10000 into `data/raw/HAM10000`.
2. Adjust `configs/config.yaml` for your hardware.
3. Train EfficientNet and ViT.
4. Compare saved evaluation metrics and confusion matrices.
5. Launch the Streamlit app with the best checkpoint.
