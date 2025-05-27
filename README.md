project:
  title: "CIFAR-10 Adversarial Training with FGSM"
  description: >
    This project demonstrates the training and evaluation of a CNN on the CIFAR-10 dataset,
    including adversarial training using FGSM to improve model robustness.

features:
  - CNN trained on CIFAR-10 dataset
  - Evaluation on clean and noisy inputs (Gaussian, salt-and-pepper, speckle)
  - FGSM adversarial attack implementation
  - Adversarial training (clean + FGSM examples)
  - Flask API for model predictions
  - Streamlit GUI for end-user interaction
  - Accepts class names (e.g., "cat") instead of numeric labels

classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

quickstart:
  steps:
    - step: "Clone the Repository"
      command: |
        git clone https://github.com/YOUR_USERNAME/cifar10-adversarial-training.git
        cd cifar10-adversarial-training

    - step: "Install Dependencies"
      command: pip install -r requirements.txt

    - step: "Export the Model from Colab"
      code: |
        model.save("model_fgsm.h5")
        from google.colab import files
        files.download("model_fgsm.h5")
      notes: "Move model_fgsm.h5 into the project directory"

    - step: "Run the Flask API"
      command: python app.py
      output: "http://127.0.0.1:5000/predict"

    - step: "Run the Streamlit GUI (Optional)"
      command: streamlit run gui_app.py

api_example:
  description: "Send an image and get prediction"
  method: POST
  endpoint: "/predict?adversarial=true&true_label=cat"
  curl: curl -X POST -F image=@cat.png "http://localhost:5000/predict?adversarial=true&true_label=cat"
  response_example:
    prediction: dog
    confidence: 0.85

structure:
  - Project.ipynb: "Training notebook (run in Colab)"
  - model_fgsm.h5: "Trained model"
  - app.py: "Flask API backend"
  - gui_app.py: "Streamlit frontend"
  - requirements.txt: "Python dependencies"
  - README.md: "Project documentation"
  - images/: "Visualization outputs"

results:
  clean_accuracy: "~70–80%"
  noise_accuracy: "Decreases with Gaussian/S&P/Speckle"
  fgsm_accuracy: "~19.5% before FGSM training; improves after"
  notes: "FGSM helps, but PGD offers stronger robustness"

future_work:
  - Use PGD or BIM for stronger adversarial defense
  - Evaluate on corrupted datasets (CIFAR-10-C)
  - Train with deeper architectures (e.g., ResNet, WideResNet)
  - Deploy online with Docker + Render or Hugging Face Spaces

references:
  - title: "Explaining and Harnessing Adversarial Examples"
    author: "Goodfellow et al."
    link: "https://arxiv.org/abs/1412.6572"

  - title: "TensorFlow FGSM Tutorial"
    link: "https://www.tensorflow.org/tutorials/generative/adversarial_fgsm"

  - title: "CIFAR-10 Dataset"
    link: "https://www.cs.toronto.edu/~kriz/cifar.html"

author:
  name: "Firas Ajengui"
  linkedin: "https://www.linkedin.com"
  github: "https://github.com/YOUR_USERNAME"
