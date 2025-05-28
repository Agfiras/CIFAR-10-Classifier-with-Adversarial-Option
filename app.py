from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model("model_fgsm.h5")

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
class_to_index = {name: idx for idx, name in enumerate(class_names)}

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((32, 32))
    img = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def fgsm_attack(model, image, label, epsilon=0.01):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.categorical_crossentropy(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adv_image = image + epsilon * signed_grad
    return tf.clip_by_value(adv_image, 0, 1)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = preprocess_image(file.read())

    # Optional FGSM
    is_adv = request.args.get("adversarial", "false") == "true"
    if is_adv:
        true_label = request.args.get("true_label", "cat").lower()
        if true_label not in class_to_index:
            return jsonify({"error": f"Invalid true_label: {true_label}"}), 400
        label_idx = class_to_index[true_label]
        label = tf.one_hot([label_idx], depth=10)
        image = fgsm_attack(model, image, label)

    pred = model.predict(image)[0]
    return jsonify({
        "prediction": class_names[int(np.argmax(pred))],
        "confidence": float(np.max(pred))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
