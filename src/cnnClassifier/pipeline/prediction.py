import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64
import os

class PredictionPipeline:
    def __init__(self, filename=None):
        self.filename = filename
        self.model = None

    def _get_model(self):
        if self.model is None:
            model_path = os.path.join("artifacts", "training", "model.h5")
            if not os.path.exists(model_path):
                model_path = os.path.join("model", "model.h5")
            self.model = load_model(model_path)
        return self.model

    def predict(self, image_data=None):
        model = self._get_model()
        target_size = (224, 224)

        if image_data is not None:
            if isinstance(image_data, str):
                if image_data.startswith("data:image"):
                    image_data = image_data.split(",", 1)[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            pil_img = pil_img.resize(target_size)
            test_image = image.img_to_array(pil_img)
        elif self.filename and os.path.exists(self.filename):
            test_image = image.load_img(self.filename, target_size=target_size)
            test_image = image.img_to_array(test_image)
        else:
            raise ValueError("No valid image data or file provided for prediction")

        # Rescale normalized pixel values matching training
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        predictions = model.predict(test_image, verbose=0)
        result = np.argmax(predictions, axis=1)

        if result[0] == 1:
            prediction = "Normal"
            return [{"image": prediction}]
        else:
            prediction = "Adenocarcinoma Cancer"
            return [{"image": prediction}]