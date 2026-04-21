# face-antispoofing-classification

Note: The notebook needs to be run on Kaggle environment.

# 🛡️ Robust Face Anti-Spoofing (FAS)
**DINOv2 Register Tokens & Multi-Head Attention Pooling**

This repository contains *deep learning* *pipeline* implementations to solve **Face Anti-Spoofing (FAS)** or *Presentation Attack Detection (PAD)* challenges. This system is designed to distinguish real faces (*live faces*) from various manipulation attacks such as printed photos, digital screens, 3D masks, and mannequins. 

This project was developed by the team **after ITB, Three Params to UGM** (Marvel Irawan, Muhammad Naufal Muzaki, and Rafsanjani).

## 🚀 Approach & Solution
To address the limited (~1.6K images) and imbalanced (*imbalance*) dataset, we abandoned the conventional architecture and used the *Vision Foundation Models* paradigm:
* **DINOv2 with Register Tokens (`dinov2_vitb14_reg`):** The *Backbone* used to clean *artifacts* is useless so the model focuses purely on skin texture or manipulation patterns.
* **Multi-Head Attention Pooling:** Replaces standard *Mean Pooling* to allow models to dynamically select face/background areas that contain *spoofing* features.
* **Supervised Contrastive & Class-Weighted Loss:** Forces the original face representation away from manipulation attack types, as well as providing heavier penalties for dealing with class imbalance.
* **Specific Anti-Spoofing Augmentations:** Uses special augmentations (such as Moiré patterns and JPEG compression) to simulate real-world *spoofing* conditions.

## ⚙️ Dependencies & Hardware
* **Library:** PyTorch, DINOv2, Albumentations, Scikit-Learn.
* **Optimization:** `OneCycleLR` scheduler and *Label Smoothing* for fast and stable convergence.
* **Hardware:** Optimized for Multi-GPU computing using `nn.DataParallel` on **2x NVIDIA T4 GPUs**.

*** Feel free to copy and adjust (e.g. add *install* steps or how to run the model) if necessary!
