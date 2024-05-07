from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/"+ imagefile.filename
    imagefile.save(image_path)
    
    models = {
    "DenseNet121": load_model("./models/DenseNet121.keras"),
    "InceptionV3": load_model("./models/InceptionV3.keras"),
    "MobileNetV1": load_model("./models/MobileNetV1.keras"),
    "Sequential CNN": load_model("./models/Sequential.keras")
}

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = {model_name: model.predict(image, verbose=0)[0].tolist() for model_name, model in models.items()}
    class_names = ['Avulsion Fracture', 'Comminuted Facture', 'Fracture Dislocation', 'Greenstick Fracture', 'Hairline Fracture', 'Impacted Fracture', 'Longitudinal Fracture', 'Oblique Fracture', 'Pathological Fracture', 'Spiral Fracture']
    results = [{"model_name": model_name, "predictions": [{"class_name": class_names[i], "probability": float(prediction[i])} for i in range(len(class_names))]} for model_name, prediction in predictions.items()]
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
       