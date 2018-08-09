import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import tensorflow as tf

app = Flask(__name__)

def get_model():
    global model, graph
    model = load_model('SLD_CNN.h5')
    print(" ---> Model loaded!")
    graph = tf.get_default_graph()
    
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

print(" ---> Loading Keras model...")
get_model()

@app.route("/sld", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    with graph.as_default(): 
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224,224))
        prediction = model.predict(processed_image).tolist()
    response = {
        'prediction': {
            '0': prediction[0][0],
            '1': prediction[0][1],
            '2': prediction[0][2],
            '3': prediction[0][3],
            '4': prediction[0][4],
            '5': prediction[0][5],
            '6': prediction[0][6],
            '7': prediction[0][7],
            '8': prediction[0][8],
            '9': prediction[0][9]
        }
    }
    return jsonify(response)

@app.route('/')
def running():
    return 'Flask is running!!!!'

if __name__ == '__main__':
  app.run()