# 🧠 Alzheimer Classification with MRI Images  
**In Memory of My Late Grandmother, Who Suffered from Alzheimer's ❤️**  

This project explores **Alzheimer's disease classification using MRI images** by leveraging advanced **deep learning** techniques. The model classifies MRI scans into four distinct stages of Alzheimer's disease, potentially aiding healthcare professionals in **early detection, monitoring disease progression, and optimizing patient care.**

---

## 📂 Dataset Overview  
**Data Source:** *Best Alzheimer MRI dataset* (99% accuracy)  
**Structure:**  
- **Training Set:** Used for model training  
- **Testing Set:** Used for evaluating performance  

Each MRI image is labeled into one of the following categories:  
- **Non-Demented:** Healthy brain images  
- **Very Mild Demented:** Early signs of Alzheimer’s  
- **Mild Demented:** Moderate progression  
- **Moderate Demented:** Advanced stage  

📊 **Class Distribution (Before SMOTE):**  
- Non-Demented: **640** samples  
- Very Mild Demented: **448** samples  
- Mild Demented: **179** samples  
- Moderate Demented: **12** samples  

⚠️ **Dataset Imbalance:**  
The dataset is highly imbalanced, which could affect model performance. To address this, we apply **SMOTE (Synthetic Minority Oversampling Technique)** to balance the class distribution.

---

## 🛠 Tools & Technologies  
- **Programming Language:** Python 🐍  
- **Deep Learning Frameworks:**  
  - **TensorFlow/Keras** – Transfer learning models  
  - **Scikit-Learn** – Data preprocessing & evaluation  
  - **NumPy & Pandas** – Data handling & analysis  
- **Computing Environment:**  
  - **Google Colab** – For model training  
  - **Kaggle API** – Direct dataset access  

---

## 🏗 Methodology  

### 1️⃣ **Baseline Training on Unbalanced Data**  
- Train an initial model (**model_1**) on the raw, imbalanced dataset using **InceptionV3, ResNet152v2, and DenseNet201**.  
- Extract **feature embeddings (Zcode)** from trained models.  

### 2️⃣ **SMOTE Application for Class Balancing**  
- **Feature-Level Balancing:** SMOTE is applied on extracted **Zcode embeddings** instead of raw images.  
- **New Balanced Dataset:** Each class is upsampled to **2560** samples, ensuring equal representation.  

### 3️⃣ **Final Model Training on Balanced Data**  
- Train a **new classifier** on the SMOTE-balanced dataset using **Zcode embeddings** from step 2.  
- Evaluate performance on original test data.  

📌 **Why Feature-Level SMOTE?**  
- Avoids **artificially creating MRI images**  
- Retains important patterns from the **original dataset**  
- **Reduces computation cost** and enhances model generalization  

---

## 📖 User Guide  

### 📌 **Notebook Structure & Runtime Instructions**  

| Part | Computing Unit | Purpose |
|------|--------------|---------|
| **Part 1** | L4 GPU | Train & test base model on unbalanced data |
| **Part 2** | L4 GPU | Extract **Zcode** (features) from model_1 |
| **Part 3** | L4 GPU | Apply **SMOTE** to Zcode features |
| **Part 4** | TPU v2-8 | Train & test classifier on balanced dataset |

### 📂 **Saved Models & Variables**  

#### ✅ **Models (Before & After SMOTE)**  
| Transfer Learning Model | Dataset | Saved Model Name |
|------------------------|------------------|------------------|
| **InceptionV3** | Before SMOTE | `Inception_model_1` |
|  | After SMOTE | `classifier_model` |
| **DenseNet201** | Before SMOTE | `DenseNet_model_1_v2` |
|  | After SMOTE | `DenseNet-classifier` |
| **ResNet152v2** | Before SMOTE | `ResNet152V2_model_1` |
|  | After SMOTE | `ResNet152V2_classifier_model` |

#### 📊 **Stored Variables for Training**  
- **Part 2 (Feature Extraction)**  
  - `Zcode_training` – Extracted training features  
  - `Zcode_training_noshuffle` – Unshuffled version  

- **Part 3 (SMOTE Application)**  
  - `combined_training_data` – Balanced dataset  
  - `combined_training_data_labels` – Corresponding labels  
  - `Zcode_testing` – Testing features for validation  
  - `combined_training_data_noshuffle` – Unshuffled version  

### ⚙️ **How to Run**  

| Part | Computing Unit | Load Models/Variables |
|------|--------------|-----------------------|
| **Part 1** | L4 GPU | No pre-loaded models required |
| **Part 2** | L4 GPU | Load model_1 (`.keras`) from **before SMOTE** |
| **Part 3** | L4 GPU | Load feature extractions (`Zcode_training`, etc.) |
| **Part 4** | TPU v2-8 | Load balanced dataset & extracted features |

📌 **Ensure that:**
- GPU/TPU selection matches each notebook part.
- Saved models (`.keras`) and variables (`.npy`) are in an accessible directory.

---

## 🎯 Key Takeaways  
✅ **Deep learning** with **MRI-based classification** can assist in detecting Alzheimer's progression.  
✅ **SMOTE-enhanced feature balancing** improves model fairness & performance.  
✅ **Structured notebook execution** ensures reproducibility.  

This project contributes towards **advancing AI-driven early detection** of Alzheimer's disease, with potential applications in medical imaging diagnostics.  

💙 *Dedicated to all families affected by Alzheimer’s.*  

