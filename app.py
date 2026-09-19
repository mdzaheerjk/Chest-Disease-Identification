from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.pipeline.prediction import PredictionPipeline

import sys

app = Flask(__name__)
CORS(app)

classifier = PredictionPipeline()

@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    try:
        os.system(f"PYTHONPATH=src {sys.executable} main.py")
        return jsonify({"message": "Model training completed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        image_b64 = request.json['image']
        if not image_b64 or len(str(image_b64).strip()) == 0:
            return jsonify({"error": "Image content is empty. Please select or upload a valid CT scan image."}), 400
        
        # Process image purely in-memory without saving any file to codebase
        result = classifier.predict(image_data=image_b64)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)