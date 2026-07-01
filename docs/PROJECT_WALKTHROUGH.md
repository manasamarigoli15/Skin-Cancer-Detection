# Skin Cancer Detection Using Deep Learning and Explainable AI

## Complete Theory and Practical Walkthrough

This document explains the full project step by step, as if you built it yourself from scratch. It combines the theory behind each part with the practical implementation used in this project.

Project folder:

```text
E:\Cancer Detection Project
```

Main goal:

Build an end-to-end deep learning application that classifies skin lesion images from the HAM10000 dataset and explains model predictions using Grad-CAM.

---

## 1. Project Objective

### Theory

Skin cancer classification from dermoscopic images is a computer vision problem. A dermoscopic image contains visual patterns such as color variation, borders, texture, symmetry, and lesion structure. Deep learning models can learn these patterns automatically from labeled images.

The project uses supervised learning. This means each training image has a known label, and the model learns to map an image to one of the skin lesion classes.

The HAM10000 dataset contains seven diagnostic categories:

| Label | Meaning |
| --- | --- |
| akiec | Actinic keratoses and intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

### Practical

We created a Python and PyTorch project with:

- HAM10000 dataset loading
- image preprocessing and augmentation
- EfficientNet baseline model
- Vision Transformer comparison model
- training and checkpointing
- evaluation metrics
- Grad-CAM visualization
- Streamlit web app
- README and configuration files

---

## 2. Project Structure

### Theory

A clean machine learning project should separate responsibilities. Dataset code, model code, training code, evaluation code, utility code, and the web app should not all be mixed into one file. This makes the project easier to understand, debug, and extend.

### Practical

The project was organized like this:

```text
Cancer Detection Project/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── docs/
│   └── PROJECT_WALKTHROUGH.md
├── logs/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── datamodule.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── metrics.py
│   ├── models/
│   │   ├── factory.py
│   │   ├── infer.py
│   │   └── predict.py
│   ├── training/
│   │   └── train.py
│   └── utils/
│       ├── checkpoint.py
│       ├── config.py
│       ├── explain.py
│       ├── gradcam.py
│       ├── logger.py
│       └── seed.py
├── requirements.txt
└── README.md
```

Important files:

- `configs/config.yaml`: all major settings
- `src/training/train.py`: training script
- `src/evaluation/evaluate.py`: evaluation script
- `src/utils/gradcam.py`: Grad-CAM implementation
- `app/streamlit_app.py`: web application
- `checkpoints/best_efficientnet.pt`: saved trained model checkpoint

---

## 3. Environment Setup

### Theory

A Python virtual environment keeps project dependencies separate from the rest of the computer. This avoids package conflicts and makes the project reproducible.

### Practical

The required packages were listed in:

```text
requirements.txt
```

Main dependencies:

- `torch`: deep learning framework
- `torchvision`: image transforms and vision utilities
- `timm`: pretrained EfficientNet and ViT models
- `pandas`: metadata handling
- `scikit-learn`: metrics and data splitting
- `matplotlib` and `seaborn`: plots and confusion matrix
- `opencv-python`: heatmap overlay for Grad-CAM
- `streamlit`: web app
- `PyYAML`: configuration loading

Typical setup commands:

```powershell
cd "E:\Cancer Detection Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## 4. Dataset Setup

### Theory

The HAM10000 dataset includes:

- image files: dermoscopic lesion images
- metadata CSV: image IDs and diagnosis labels

The model does not read the CSV alone. It needs both the CSV labels and the actual image files.

### Practical

The dataset was uploaded under:

```text
data/raw/
```

Actual detected structure:

```text
data/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
├── hmnist_28_28_L.csv
├── hmnist_28_28_RGB.csv
├── hmnist_8_8_L.csv
└── hmnist_8_8_RGB.csv
```

At first, the config expected:

```text
data/raw/HAM10000/HAM10000_metadata.csv
```

But your dataset was actually here:

```text
data/raw/HAM10000_metadata.csv
```

So we updated `configs/config.yaml` to:

```yaml
data:
  root_dir: data/raw
  metadata_csv: data/raw/HAM10000_metadata.csv
  image_dirs:
    - data/raw/HAM10000_images_part_1
    - data/raw/HAM10000_images_part_2
    - data/raw/images
```

Then we verified the dataset successfully:

```text
metadata rows: 10015
labels: akiec, bcc, bkl, df, mel, nv, vasc
```

This confirmed that the metadata and image paths were correctly connected.

---

## 5. Configuration File

### Theory

A configuration file stores experiment settings outside the code. This lets us change image size, batch size, epochs, model names, and paths without editing Python logic.

### Practical

The main config file is:

```text
configs/config.yaml
```

Important settings:

```yaml
project:
  seed: 42
  device: auto

data:
  image_size: 224
  batch_size: 32
  val_size: 0.15
  test_size: 0.15

training:
  epochs: 15
  learning_rate: 0.0003
  weight_decay: 0.0001
  use_weighted_loss: true
  mixed_precision: true
  early_stopping_patience: 5

models:
  efficientnet:
    name: efficientnet_b0
    pretrained: true
  vit:
    name: vit_b_16
    pretrained: true
```

`device: auto` means the code uses GPU if available, otherwise CPU.

In your run, the log showed:

```text
Using device: cpu
```

So training was done on CPU.

---

## 6. Image Preprocessing and Augmentation

### Theory

Deep learning models require images to have the same size and numeric format.

Preprocessing includes:

1. resizing images to a fixed size
2. converting images to tensors
3. normalizing pixel values

Data augmentation creates modified versions of training images to improve generalization. It helps the model avoid memorizing exact images.

Common augmentations:

- horizontal flip
- vertical flip
- rotation
- brightness/contrast/saturation changes

### Practical

Implemented in:

```text
src/data/dataset.py
```

Training transforms:

```python
transforms.Resize((image_size, image_size))
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.RandomRotation(20)
transforms.ColorJitter(...)
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Validation, test, and app inference use only deterministic preprocessing:

```python
transforms.Resize((image_size, image_size))
transforms.ToTensor()
transforms.Normalize(...)
```

This is important because evaluation should be consistent and not randomly modify images.

---

## 7. Data Splitting

### Theory

The dataset is split into:

- training set: used to learn model weights
- validation set: used during training to choose the best checkpoint
- test set: used after training to measure final performance

Stratified splitting keeps class proportions similar in each split. This matters because HAM10000 is imbalanced, especially with many `nv` samples and fewer rare disease classes.

### Practical

Implemented in:

```text
src/data/datamodule.py
```

The config uses:

```yaml
val_size: 0.15
test_size: 0.15
```

So the approximate split is:

- 70% training
- 15% validation
- 15% test

A weighted sampler was also used during training so rare classes are sampled more often.

---

## 8. Model 1: EfficientNet Baseline

### Theory

EfficientNet is a convolutional neural network designed to balance accuracy and computational efficiency. It scales depth, width, and resolution in a systematic way.

EfficientNet is a good baseline because:

- it performs well on image classification
- it is lighter than many large CNNs
- it works well with transfer learning
- it trains faster than large transformer models

Transfer learning means we start from a model pretrained on ImageNet and replace the final classification layer for our seven HAM10000 classes.

### Practical

Implemented in:

```text
src/models/factory.py
```

EfficientNet model config:

```yaml
efficientnet:
  name: efficientnet_b0
  pretrained: true
```

Created using `timm`:

```python
timm.create_model(
    model_config["name"],
    pretrained=True,
    num_classes=num_classes,
)
```

Training command:

```powershell
python -m src.training.train --model efficientnet
```

Your training started successfully and downloaded pretrained weights:

```text
model.safetensors: 100% downloaded
```

---

## 9. Model 2: Vision Transformer

### Theory

A Vision Transformer, or ViT, treats an image as a sequence of patches. Instead of using convolution filters, it uses self-attention to learn relationships between different image regions.

ViT is useful for comparison because it represents a different deep learning approach from CNNs.

CNNs such as EfficientNet learn local spatial patterns naturally. ViTs learn global relationships through attention, but they often need more data and compute.

### Practical

ViT model config:

```yaml
vit:
  name: vit_b_16
  pretrained: true
```

Training command:

```powershell
python -m src.training.train --model vit
```

Because your machine used CPU, ViT training may be much slower than EfficientNet. For practical demonstration, EfficientNet is the better model to train first.

---

## 10. Training Process

### Theory

Training means the model repeatedly sees image batches, predicts labels, calculates error, and updates weights using backpropagation.

Main training components:

- loss function: measures prediction error
- optimizer: updates model weights
- learning rate: controls update size
- epochs: full passes over the training dataset
- validation metrics: measure progress on unseen validation data

This project uses:

- CrossEntropyLoss for multi-class classification
- AdamW optimizer
- CosineAnnealingLR learning-rate scheduler
- weighted loss for class imbalance
- mixed precision when CUDA is available
- early stopping
- checkpointing

### Practical

Training code:

```text
src/training/train.py
```

Command used:

```powershell
python -m src.training.train --model efficientnet
```

Your log showed:

```text
2026-06-30 22:10:00 | INFO | train.efficientnet | Using device: cpu
```

Epoch 1 result:

```text
train_loss=0.8691
val_acc=0.3160
val_f1=0.3079
val_auc=0.8725
Saved new best checkpoint: checkpoints\best_efficientnet.pt
```

Epoch 2 result:

```text
train_loss=0.4278
val_acc=0.3846
val_f1=0.3618
val_auc=0.9140
Saved new best checkpoint: checkpoints\best_efficientnet.pt
```

This means:

- training loss decreased
- validation F1 improved
- ROC-AUC improved
- the best checkpoint was saved successfully

You later closed the command window, but the checkpoint had already been saved.

Verified checkpoint:

```text
checkpoints\best_efficientnet.pt
```

---

## 11. Checkpointing

### Theory

A checkpoint stores model weights and training information. This allows us to use the model later without retraining from scratch.

A good project saves the best model based on validation performance, not just the last epoch.

### Practical

Implemented in:

```text
src/utils/checkpoint.py
```

The project saves:

- model weights
- optimizer state
- epoch number
- metrics
- config
- class-to-index mapping

Saved checkpoint path:

```text
checkpoints/best_efficientnet.pt
```

This is the file used by evaluation and the Streamlit app.

---

## 12. Evaluation Metrics

### Theory

Accuracy alone is not enough for medical image classification because datasets are often imbalanced.

This project evaluates:

| Metric | Meaning |
| --- | --- |
| Accuracy | Overall percentage of correct predictions |
| Precision | Of predicted positives, how many were correct |
| Recall | Of actual positives, how many were found |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to separate classes across thresholds |
| Confusion Matrix | Shows correct and incorrect predictions per class |

Macro averaging treats all classes equally, which is useful for imbalanced datasets.

### Practical

Evaluation code:

```text
src/evaluation/evaluate.py
src/evaluation/metrics.py
```

Command:

```powershell
python -m src.evaluation.evaluate --model efficientnet --checkpoint checkpoints/best_efficientnet.pt
```

Expected outputs:

```text
reports/efficientnet_metrics.json
reports/efficientnet_classification_report.csv
reports/figures/efficientnet_confusion_matrix.png
```

For ViT, after training:

```powershell
python -m src.evaluation.evaluate --model vit --checkpoint checkpoints/best_vit.pt
```

---

## 13. Grad-CAM Explainability

### Theory

Grad-CAM stands for Gradient-weighted Class Activation Mapping. It highlights image regions that influenced a model prediction.

For a CNN, Grad-CAM uses gradients flowing into the final convolutional layer. Areas with strong influence are shown as hotter colors on a heatmap.

This is useful in medical AI because users should not only see the predicted class but also inspect whether the model focused on the lesion area.

Important limitation:

Grad-CAM does not prove the model is medically correct. It only shows which regions influenced the model.

### Practical

Implemented in:

```text
src/utils/gradcam.py
```

For EfficientNet, the target layer is:

```python
model.conv_head
```

For ViT, patch-token activations are reshaped into a spatial grid.

The app overlays the heatmap on the uploaded lesion image.

---

## 14. Streamlit Web App

### Theory

A web app makes the model usable by non-programmers. Instead of running Python commands, a user can upload an image and see the prediction, confidence, explanation, and heatmap.

### Practical

App file:

```text
app/streamlit_app.py
```

Run command:

```powershell
streamlit run app/streamlit_app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

In the app:

1. Keep model as `efficientnet`.
2. Keep checkpoint path as:

```text
checkpoints/best_efficientnet.pt
```

3. Upload one lesion image, for example from:

```text
data/raw/HAM10000_images_part_1/
```

or:

```text
data/raw/HAM10000_images_part_2/
```

Example file type:

```text
ISIC_0027419.jpg
```

The app displays:

- uploaded image
- predicted class
- confidence score
- Grad-CAM heatmap
- class probability chart
- short explanation

---

## 15. AI-Generated Explanation

### Theory

An explanation should be understandable to humans. It should state the predicted class, confidence, and what the heatmap means.

Because this is a medical-related application, the explanation must also include a caution that the output is not a medical diagnosis.

### Practical

Implemented in:

```text
src/utils/explain.py
```

The explanation includes:

- predicted class name
- confidence percentage
- statement about heatmap regions
- medical safety disclaimer

Example style:

```text
The model predicts Melanoma (mel) with 82.4% confidence. The heatmap highlights image regions that most influenced the score...
```

---

## 16. Commands Used in This Project

### Check dataset layout

```powershell
Get-ChildItem -Recurse -Depth 3 -Force data\raw
```

### Verify metadata path

```powershell
Test-Path -LiteralPath 'data\raw\HAM10000_metadata.csv'
```

### Train EfficientNet

```powershell
python -m src.training.train --model efficientnet
```

### Evaluate EfficientNet

```powershell
python -m src.evaluation.evaluate --model efficientnet --checkpoint checkpoints/best_efficientnet.pt
```

### Run Streamlit app

```powershell
streamlit run app/streamlit_app.py
```

### Train ViT, optional and slower on CPU

```powershell
python -m src.training.train --model vit
```

### Evaluate ViT

```powershell
python -m src.evaluation.evaluate --model vit --checkpoint checkpoints/best_vit.pt
```

---

## 17. What Happened During Your Run

You ran:

```powershell
python -m src.training.train --model efficientnet
```

The model used CPU:

```text
Using device: cpu
```

The pretrained EfficientNet weights downloaded successfully.

The first two epochs completed and saved the best model checkpoint:

```text
checkpoints\best_efficientnet.pt
```

You closed the command window after training had already saved a usable model checkpoint.

We checked and confirmed:

```text
best_efficientnet.pt exists
```

So the next practical step is evaluation:

```powershell
python -m src.evaluation.evaluate --model efficientnet --checkpoint checkpoints/best_efficientnet.pt
```

Then run the app:

```powershell
streamlit run app/streamlit_app.py
```

---

## 18. How to Explain This Project in Your Own Words

You can say:

```text
I built a deep learning-based skin lesion classification system using the HAM10000 dataset. I organized the project into separate modules for data loading, preprocessing, model creation, training, evaluation, explainability, and deployment. I used EfficientNet as the baseline model and included Vision Transformer for comparison. Images were resized, normalized, and augmented during training. I used stratified train-validation-test splitting and handled class imbalance with weighted sampling and weighted loss. The trained model was evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix. I added Grad-CAM to visualize which image regions influenced predictions. Finally, I built a Streamlit web app where users can upload a lesion image and view the predicted class, confidence score, Grad-CAM heatmap, and explanation.
```

---

## 19. Important Limitations

This project is educational and experimental.

Limitations:

- It is not a medical diagnostic tool.
- HAM10000 is imbalanced, so rare classes are harder to learn.
- CPU training is slow.
- A model trained for only a few epochs is useful for demonstration but not final performance.
- Grad-CAM is an explanation aid, not proof of correctness.
- Real clinical use would require expert validation, external testing, fairness checks, and regulatory approval.

---

## 20. Recommended Next Steps

1. Run EfficientNet evaluation.
2. Open the Streamlit app and test several images.
3. Train EfficientNet for more epochs if time allows.
4. Train ViT only if you have enough time or GPU support.
5. Compare EfficientNet and ViT metrics.
6. Include screenshots of the app, heatmap, confusion matrix, and metrics in your final report or presentation.

---

## 21. Quick Final Checklist

- Dataset loaded correctly: yes
- Metadata rows found: 10015
- Classes found: 7
- EfficientNet training started: yes
- Checkpoint saved: yes
- Evaluation command ready: yes
- Streamlit app ready: yes
- Grad-CAM implemented: yes
- README included: yes
- Project walkthrough document created: yes

