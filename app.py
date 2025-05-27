from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)  # Load the pre-trained model
model = tf.keras.models.load_model("model_fgsm.h5") # Ensure the model is in evaluation mode

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(img_bytes): 
    img = Image.open(io.BytesIO(img_bytes)).resize((32, 32)) # Resize to match model input size
    img = np.array(img).astype('float32') / 255.0 # Normalize the image
    return np.expand_dims(img, axis=0) # Add batch dimension

def fgsm_attack(model, image, label, epsilon=0.01): 
    with tf.GradientTape() as tape: 
        tape.watch(image) # Watch the input image for gradients
        prediction = model(image) # Get model prediction
        loss = tf.keras.losses.categorical_crossentropy(label, prediction) # Compute loss
    gradient = tape.gradient(loss, image) 
    signed_grad = tf.sign(gradient)
    adv_image = image + epsilon * signed_grad
    return tf.clip_by_value(adv_image, 0, 1)

@app.route("/predict", methods=["POST"]) # Endpoint for image prediction
def predict(): 
    file = request.files["image"] # Get the uploaded image file
    image = preprocess_image(file.read()) # Preprocess the image

    is_adv = request.args.get("adversarial", "false") == "true" # Check if adversarial attack is requested
    if is_adv:
        label_idx = int(request.args.get("true_label", "0"))
        label = tf.one_hot([label_idx], depth=10)
        image = fgsm_attack(model, image, label)

    pred = model.predict(image)[0]
    label = class_names[int(np.argmax(pred))]
    conf = float(np.max(pred))

    return jsonify({"prediction": label, "confidence": conf})

if __name__ == "__main__":
    app.run(debug=True) # Run the Flask app
