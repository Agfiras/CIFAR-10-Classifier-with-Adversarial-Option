
---
# 🧠 CIFAR-10 Adversarial Training with FGSM

This project demonstrates how to build and evaluate a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset, with improved robustness through **FGSM (Fast Gradient Sign Method)** adversarial training.

Deployed live using **Hugging Face Spaces + Streamlit** 🚀

🔗 **[Try the Live Demo]([https://huggingface.co/spaces YOUR_USERNAME/cifar10-fgsm](https://huggingface.co/spaces/3llisa/CIFAR-10-Classifier-with-Adversarial-Option))**
---
## 📚 Overview
### 🎯 Objectives

- Train a CNN model on CIFAR-10
- Evaluate robustness under both natural and adversarial perturbations
- Implement **adversarial training** using FGSM-generated examples
- Provide an interactive demo via **Streamlit** on Hugging Face Spaces

---

## 🗂️ Project Structure

```
cifar10-adversarial-training/
├── Project.ipynb            # Colab notebook for training
├── model_fgsm.h5            # Trained model exported from Colab
├── app.py                   # Streamlit app for Hugging Face Spaces
├── requirements.txt         # Dependencies for deployment
├── README.md                # Project documentation
└── images/                  # Visual results (optional)
```

---

## 🚀 How to Use Locally

### 1. Clone the repository

```bash
git clone https://github.com/Agfiras/CIFAR-10-Classifier-with-Adversarial-Option.git
cd cifar10-adversarial-training
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app locally

```bash
streamlit run app.py
```

---

## 📦 Deploying to Hugging Face Spaces

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Create a new Space → choose **Streamlit**
3. Upload these files:
   - `app.py`
   - `model_fgsm.h5`
   - `requirements.txt`
4. Commit and deploy

> The model file must be under 5GB. If larger, compress or quantize.

---

## 🔐 FGSM Adversarial Attack

FGSM perturbs an input image in the direction of greatest model error:

\[
x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
\]

This creates adversarial samples that:
- Look similar to humans 👀
- Fool the model with high confidence 🤖

---

## 🧠 CIFAR-10 Class Labels

```
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

These class names are used in the app when applying FGSM attacks.

---

## 📊 Features in the Demo

- Upload a CIFAR-10 image (or any 32×32 RGB image)
- Toggle FGSM attack on/off
- Choose the true label (required for adversarial generation)
- View model prediction and confidence

---

## 📌 Future Improvements

- Add PGD (multi-step adversarial attack)
- Deploy as a Gradio or web API
- Extend to CIFAR-100 or TinyImageNet
- Add adversarial detection or confidence-based rejection

---

## 👤 Author

**Firas Ajengui**  
University of Szeged  
Master’s Student in Computer Science  
Trained on Google Colab, deployed with Streamlit on Hugging Face Spaces
