import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("CIFAR-10 Classifier with FGSM Adversarial Attack")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_fgsm.h5")

model = load_model()

# CIFAR-10 class names
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# Preprocess image
def preprocess(img):
    img = img.resize((32, 32))
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# FGSM adversarial attack
def create_adversarial_example(image, label_idx, epsilon=0.01):
    label = tf.one_hot([label_idx], depth=10)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.categorical_crossentropy(label, prediction)

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adv_img = image + epsilon * signed_grad
    return tf.clip_by_value(adv_img, 0, 1)

uploaded_file = st.file_uploader("Upload a CIFAR-10 image", type=["jpg", "jpeg", "png"])
epsilon = st.slider("FGSM epsilon", min_value=0.0, max_value=0.1, value=0.01, step=0.005)
use_fgsm = st.checkbox("Apply FGSM Adversarial Attack")
# true_label_idx = st.selectbox("True label index (required for FGSM)", list(range(10)), index=3)
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

selected_label = st.selectbox("True class (required for FGSM)", class_names, index=3)
true_label_idx = class_names.index(selected_label)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False)
    input_img = preprocess(image)

    if use_fgsm:
        adv_img = create_adversarial_example(tf.convert_to_tensor(input_img), true_label_idx, epsilon)
        prediction = model.predict(adv_img)[0]
        st.image(adv_img[0].numpy(), caption="Adversarial Image")
    else:
        prediction = model.predict(input_img)[0]

    pred_label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    st.markdown(f"### Prediction: **{pred_label}** ({confidence:.2f}%)")
