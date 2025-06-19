# Animal Image Classification using CNN and Transfer Learning with DenseNet121

This project implements an image classification model using Convolutional Neural Networks (CNN) with transfer learning via DenseNet121. The model is trained on the Animal-10 dataset, which contains 10 classes of animals. It includes steps for data augmentation, class balancing, model fine-tuning, and performance evaluation. Achieves over 95% validation accuracy, demonstrating strong generalization and effectiveness in real-world image recognition tasks.

---

## 📁 Dataset

- **Name**: Animal-10
- **Source**: [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- **Classes**:
  - `butterfly`, `horse`, `spider`, `squirrel`, `cow`, `chicken`, `dog`, `sheep`, `cat`, `elephant`

---

## 🧠 Model Overview

- **Architecture**: Convolutional Neural Network (CNN)
- **Base Model**: DenseNet121 (pre-trained on ImageNet)
- **Techniques**:
  - Data Augmentation
  - Transfer Learning
  - Fine-tuning (last 10 layers)
  - Batch Normalization & Dropout for regularization
- **Optimizer**: Adam (`learning_rate=1e-5`)
- **Loss Function**: Categorical Crossentropy

---

## 🏁 Results

| Metric              | Value     |
|---------------------|-----------|
| Training Accuracy   | 95.86%    |
| Validation Accuracy | 95.97%    |
| Final Training Loss | 0.2477    |

---

## 📦 Files Included

- `Klasifikasi_Gambar_Hewan_CNN_Animal10.ipynb`: Main training notebook
- `model_animal10.keras`: Trained Keras model (native format)
- `requirements.txt`: List of required dependencies

---

## 🔧 Installation

1. Clone this repository:

```bash
git clone https://github.com/wilmanurfatimah/animal10-cnn-classification.git
cd animal10-cnn-classification

2. Install dependencies:
pip install -r requirements.txt

3. Run the notebook:
jupyter notebook Klasifikasi_Gambar_Hewan_CNN_Animal10.ipynb


✍️ Author
Wilma Nur Fatimah
Machine Learning Enthusiast

