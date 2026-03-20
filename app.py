from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model("mnist_model.h5")


@app.route("/")
def home():
    return "API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = request.json
        image = np.array(data).reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(image)

        return jsonify({
            "prediction": int(np.argmax(prediction))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    print("Starting app...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)