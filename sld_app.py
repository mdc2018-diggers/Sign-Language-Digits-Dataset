import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.applications.mobilenet import preprocess_input
from flask import request
from flask import jsonify
from flask import Flask
import tensorflow as tf

from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

app = Flask(__name__)

def get_model():
    global model, graph
    with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
        model = load_model('SLD_CNN.h5')
    print(" ---> Model loaded!")
    graph = tf.get_default_graph()
    
def preprocess_image(image, target_size):
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def get_label(index):
    correct_label = [9, 0, 7, 6, 1, 8, 4, 3, 2, 5]
    return correct_label[index]

print(" ---> Loading Keras model...")
get_model()

@app.route("/sld", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    with graph.as_default(): 
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(128, 128))
        processed_image = preprocess_input(processed_image)
        prediction = model.predict(processed_image).tolist()
    response = {
        'prediction': {
            '0': prediction[0][get_label(0)],
            '1': prediction[0][get_label(1)],
            '2': prediction[0][get_label(2)],
            '3': prediction[0][get_label(3)],  
            '4': prediction[0][get_label(4)],
            '5': prediction[0][get_label(5)],
            '6': prediction[0][get_label(6)],
            '7': prediction[0][get_label(7)],
            '8': prediction[0][get_label(8)],
            '9': prediction[0][get_label(9)]
        }
    }
    return jsonify(response)

@app.route('/')
def running():
    return 'Flask is running!!!!'

if __name__ == '__main__':
  app.run()