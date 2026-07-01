# 🧾🤖 Real vs. AI-Generated Receipt Detection

**Author:** Guntaka Pranav Nadh Reddy  
**Track:** Software Engineering — Internship Project  
**Organization:** Accenture — SAP Expense Enhancement Initiative  

---

## 📌 Project Overview
This repository contains the deep learning pipeline developed to evaluate and classify genuine, physically captured receipt images versus synthetically generated (AI-rendered) receipts. 

The primary objective is to support **expense-fraud risk reduction** within enterprise systems. During development, Explainable AI (Grad-CAM) was utilized to diagnose "shortcut learning," leading to the engineering of a highly robust, adversarial data augmentation pipeline. This forces the model to evaluate genuine structural authenticity rather than relying on superficial image artifacts.

---

## ⚙️ System Architecture & Process Flow
The project is built on a modular, end-to-end computer vision pipeline using PyTorch. The workflow is divided into five core stages:

1. **Data Ingestion & Standardization:** * Raw images are loaded, cleaned, and standardized to JPEG format.
   * Images are aggressively resized to `448 × 448` pixels using LANCZOS resampling to preserve fine-grained text and artifact details without introducing aliasing.
2. **Leakage-Free Splitting:** * A group-level splitting algorithm partitions the data (70% Train, 14.9% Val, 15.1% Test). This ensures that an original image and any augmented siblings stay strictly within the same partition, preventing data leakage.
3. **Anti-Shortcut Augmentation (Training Only):**
   * The training set is passed through a custom PyTorch `Transforms` pipeline (Random Rotation, Color Jitter, Gaussian Blur, Random Erasing) to artificially degrade the images and destroy superficial "shortcuts" (like background textures and lighting).
4. **Feature Extraction & Classification (ResNet50):** * The core model utilizes a ResNet50 architecture. The deep residual layers act as feature extractors, routing to a fully connected layer optimized for binary classification (Real vs. AI). Class weights are applied during loss calculation to handle the slight class imbalance.
5. **Explainability & Validation (Grad-CAM):**
   * Post-training, Grad-CAM (Gradient-weighted Class Activation Mapping) is applied to the final convolutional layers. This generates heatmaps over the test images, verifying that the model's attention is focused on the actual receipt content rather than the background or image borders.

---

## 📊 Dataset Statistics
The dataset is structured as a binary classification corpus. 
* **Real Receipts Data Sources:** SOIRE, WildReceipt, and expressExpense.
* **AI Receipts Data Sources:** GPT-4o and DALL-E.

| Class | Image Count | Unique Source Groups | Class Weight Applied |
| :--- | :--- | :--- | :--- |
| **Real Receipts** | 1,950 | 1,075 | 0.9062 |
| **AI-Generated Receipts** | 1,584 | 808 | 1.1155 |
| **Total** | **3,534** | **1,883** | — |

---

## 🛑 The "Shortcut Learning" Discovery
During the initial training of the ResNet50 architecture, the model achieved >99% validation accuracy within the first two epochs. Because real-world document forgery detection rarely converges this quickly, it was treated as a diagnostic red flag. 

Using **Grad-CAM** to visualize the network's attention layers, three distinct failure modes were identified:

1. **Background Context Bias (Mode A):** The model ignored receipt text and focused on backgrounds (e.g., wood textures = "Real", plain backgrounds = "AI").
2. **Capture-Hardware Artifact Bias (Mode B):** The model acted as a camera-type classifier (e.g., uniform lighting = "AI", flash glares/blur = "Real").
3. **Macro-Layout Shortcut (Mode C):** The model memorized the overall structural formatting of templates rather than detecting fine-grained AI text irregularities.

## 🛠️ The Remediation Pipeline
To neutralize these shortcuts, an aggressive, targeted data augmentation pipeline was engineered in PyTorch. 

```python
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomRotation(degrees=45, fill=255),               # Breaks background & framing bias
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), # Kills lighting/capture hardware bias
    T.GaussianBlur(kernel_size=5),                        # Kills render-sharpness bias
    T.ToTensor(),
    T.RandomErasing(p=0.3, scale=(0.02, 0.1)),            # Breaks macro-layout memorization
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
