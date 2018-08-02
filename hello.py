import keras
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)
app.debug = True

@app.route('/')
def running():
    return 'Flask is running!'

if __name__ == '__main__':
  app.run()