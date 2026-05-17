# Part 2: Computer Vision — Manufacturing Defect Classification

**BITSOM BA - Module 5 | CNN Prototype**

## Problem Statement
Classify product surface images into one of four defect categories using a Convolutional Neural Network.

## Problem Type: Image Classification
Each image belongs to exactly one class — no bounding boxes or pixel-level segmentation needed. A CNN classifier is the right tool here.

## Dataset
| Property | Value |
|---|---|
| Total images | 480 |
| Classes | dent, normal, scratch, stain |
| Images per class | 120 (balanced) |
| Image size | 96×96 px → resized to 64×64 |
| Color | RGB |

No class imbalance — each class has exactly 120 images.

## Repository Structure
```
part-2-cnn-computer-vision/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    └── confusion_matrix.png
```

## Model Architecture
```
Input (64×64×3)
    → Data Augmentation (flip, rotate, zoom)
    → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
    → Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
    → Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
    → Flatten
    → Dense(128, ReLU) → Dropout(0.4)
    → Dense(4, Softmax)
```

## CNN Concept Explanation

### What is convolution?
A convolution slides a small filter (e.g. 3×3) across the image and computes dot products at every position. This detects local patterns like edges, curves, and textures. Unlike a regular dense layer that looks at every pixel independently, a convolution layer understands *spatial relationships* between neighboring pixels.

### Why is pooling used?
MaxPooling reduces the spatial size of feature maps (e.g. 32×32 → 16×16) by keeping only the strongest activation in each region. This makes the model faster, uses less memory, and makes it robust to small shifts or translations in the image — a scratch in the top-left vs top-center still gets detected.

### Why is ReLU commonly used in CNNs?
ReLU (Rectified Linear Unit) outputs `max(0, x)` — it sets negative activations to zero and keeps positive ones. It's preferred in CNNs because:
1. It doesn't suffer from the vanishing gradient problem (unlike sigmoid/tanh)
2. It's computationally cheap
3. It creates sparsity — not all neurons fire, which improves efficiency

### Why are CNNs better than regular feed-forward networks for images?
A regular dense network would treat each pixel as an independent input, losing all spatial structure. CNNs use:
- **Local receptive fields** — each filter only looks at a small patch
- **Weight sharing** — the same filter is reused across the entire image (far fewer parameters)
- **Hierarchical feature learning** — early layers learn edges, deeper layers learn shapes/textures/objects

A 64×64×3 image has 12,288 raw inputs. A dense network would need millions of parameters just for the first layer. A CNN solves this efficiently.

## Business Use Case: Manufacturing Quality Control

**Domain:** Manufacturing / Industrial Inspection

Assembly lines produce thousands of products per hour — manual visual inspection is slow, expensive, and inconsistent. A CNN-based defect classifier can:

- **Replace manual inspection**: Cameras mounted on the line feed images to the model in real-time
- **Classify defects instantly**: Flags dents, scratches, and stains before packaging
- **Reduce waste**: Catches defects early rather than at the end of production
- **Scale easily**: One model runs on multiple production lines simultaneously

**Example companies using this:** Tesla (panel inspection), Samsung (screen defect detection), Foxconn (PCB inspection).

The model in this project directly mirrors this use case — trained on labeled defect images to predict the type of surface damage on a product.

## How to Run
1. Open `notebook.ipynb` in Google Colab
2. Upload the Part 2 dataset zip when prompted
3. Run all cells in order
4. Results are saved to `results/` and `sample_predictions/`

## Requirements
```
tensorflow>=2.10
scikit-learn>=1.0
numpy>=1.21
matplotlib>=3.5
seaborn>=0.11
Pillow>=9.0
```
