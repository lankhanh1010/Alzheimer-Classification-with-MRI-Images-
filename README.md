# 🧠 Alzheimer Classification with MRI Images  
In Memory of My Late Grandmother, who battled Alzheimer's

This project explored Alzheimer's disease classification using MRI images by leveraging advanced deep learning techniques. The model classified MRI scans into four distinct stages of Alzheimer's disease, potentially aiding healthcare professionals in early detection, monitoring disease progression, and optimizing patient care.

---

## 🛠 Tools & Technologies  
- Programming Language: Python
- Deep Learning Frameworks:
  - TensorFlow/Keras – Transfer learning models  
  - Scikit-Learn – Data preprocessing & evaluation  
  - NumPy & Pandas – Data handling & analysis  
- Computing Environment: 
  - Google Colab – For model training  
---

## 📂 Dataset Overview  
### 1️⃣ Data Source:**
- Best Alzheimer MRI dataset (99% accuracy) 
https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy

- Each MRI image is labeled into one of the following categories:  
  - Non-Demented: Healthy brain images  
  - Very Mild Demented: Early signs of Alzheimer’s  
  - Mild Demented: Moderate progression  
  - Moderate Demented: Advanced stage  

### 2️⃣ **Class Distribution (Before SMOTE):**  
- Non-Demented: 640 samples  
- Very Mild Demented: 448 samples  
- Mild Demented: 179 samples  
- Moderate Demented: 12 samples
  
![image](https://github.com/user-attachments/assets/db49491b-bf7a-403f-9461-461c00f80bc1)


### 3️⃣ **Dataset Imbalance:**  
The dataset was highly imbalanced, which could affect model performance. To address this, we applied SMOTE (Synthetic Minority Oversampling Technique) to balance the class distribution.

![image](https://github.com/user-attachments/assets/b47d721c-16f9-49e4-a004-ae90f2794026)


![image](https://github.com/user-attachments/assets/1d969ed2-bdbd-4c22-aeda-98f1879c369b)

---

## 🏗 Methodology  

### 1️⃣ **Baseline Training on Unbalanced Data**  
- The first step involved training a base model using InceptionV3, ResNet152v2, and DenseNet201 with the given imbalance dataset.
- Extracted feature embeddings (Zcode) from the trained models, forming Zcode_training and Zcode_testing


### 2️⃣ **SMOTE Application for Class Balancing**  
- Instead of generating synthetic images, SMOTE was applied to Zcode_training and Zcode_testing to balance class distributions.
- The training dataset was adjusted to ensure that each category contained the same number of samples (2560 samples).


### 3️⃣ **Final Model Training on Balanced Data**  
- The final classifier was trained using the SMOTE-balanced dataset with extracted features from step 2.
- Evaluated performance using test data without SMOTE to measure real-world applicability.

---

## 📖 User Guide  

### 📌 **Notebook Structure & Runtime Instructions**  

| Part | Computing Unit | Purpose |
|------|--------------|---------|
| **Part 1** | L4 GPU | Train & test base model on unbalanced data |
| **Part 2** | L4 GPU | Extract Zcode from model_1 |
| **Part 3** | L4 GPU | Apply SMOTE to Zcode features |
| **Part 4** | TPU v2-8 | Train & test classifier on balanced dataset |

### 📂 **Saved Models & Variables**  

#### 1️⃣ **Models (Before & After SMOTE)**  
| Transfer Learning Model | Dataset | Saved Model Name |
|------------------------|------------------|------------------|
| **InceptionV3** | Before SMOTE | `Inception_model_1` |
|  | After SMOTE | `classifier_model` |
| **DenseNet201** | Before SMOTE | `DenseNet_model_1_v2` |
|  | After SMOTE | `DenseNet-classifier` |
| **ResNet152v2** | Before SMOTE | `ResNet152V2_model_1` |
|  | After SMOTE | `ResNet152V2_classifier_model` |

#### 2️⃣ **Stored Variables for Training**  
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
| **Part 2** | L4 GPU | Load model_1 (`.keras`) from before SMOTE |
| **Part 3** | L4 GPU | Load feature extractions (`Zcode_training`, etc.) |
| **Part 4** | TPU v2-8 | Load balanced dataset & extracted features |

📌 **Ensure that:**
- GPU/TPU selection matches each notebook part.
- Saved models (`.keras`) and variables (`.npy`) are in an accessible directory.

This project contributes towards advancing AI-driven early detection of Alzheimer's disease, with potential applications in medical imaging diagnostics.  

*Dedicated to all families affected by Alzheimer’s.*  

