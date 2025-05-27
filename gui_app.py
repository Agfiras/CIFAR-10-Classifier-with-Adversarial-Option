import streamlit as st
import requests

st.title("CIFAR-10 Classifier with Adversarial Option") # Streamlit app title

file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"]) # File uploader for images
attack = st.checkbox("Apply FGSM Adversarial Attack") # Checkbox to apply FGSM attack
true_label = st.selectbox("True label (for FGSM)", list(range(10))) # Dropdown to select true label for FGSM attack

if file:
    st.image(file, caption="Uploaded Image") # Display the uploaded image
    files = {"image": file.getvalue()} # Prepare the file for the request
    url = f"http://localhost:5000/predict?adversarial={attack}&true_label={true_label}" # Construct the URL with parameters
    response = requests.post(url, files=files) # Send the POST request to the Flask app
    result = response.json() # Parse the JSON response
    st.success(f"Prediction: {result['prediction']} ({result['confidence']*100:.2f}%)") # Display the prediction result
