import streamlit as st
import requests

st.set_page_config(page_title="CIFAR-10 FGSM Demo", layout="centered")

st.title("🧠 CIFAR-10 Classifier with FGSM Adversarial Option")

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# File uploader
uploaded_file = st.file_uploader("Upload an image (32x32 CIFAR-10 format)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# FGSM toggle
attack = st.checkbox("Apply FGSM Adversarial Attack")

# Label input
true_label = st.selectbox("True Label (required if FGSM is selected)", class_names)

# Submit button
if uploaded_file and st.button("Predict"):
    files = {"image": uploaded_file.getvalue()}

    # 🔗 UPDATE this with your Flask API URL
    base_url = "https://your-flask-api.onrender.com/predict"
    params = {
        "adversarial": str(attack).lower(),
        "true_label": true_label.lower()
    }

    try:
        response = requests.post(base_url, params=params, files=files)
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Prediction: {result['prediction']} ({result['confidence']*100:.2f}%)")
        else:
            st.error(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"⚠️ Connection failed: {e}")
