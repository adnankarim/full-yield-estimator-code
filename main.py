from flask import Flask, request, jsonify,send_file

from app.torch_utils import get_prediction
from app.torch_utils_mango import get_prediction_mango
import requests
from PIL import Image
import os
import io
import time
from flask_ngrok import run_with_ngrok
from base64 import encodebytes
import json
app = Flask(__name__)
run_with_ngrok(app)
import base64
from flask_cors import CORS, cross_origin
from multiprocessing import Value

counter = Value('i', 0)
cors = CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    pil_img=pil_img.resize((250,291))
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img
app.config['CORS_HEADERS'] = 'Content-Type'
@cross_origin()
@app.route('/predicttree', methods=['POST'])
def predicttree():
    if request.method == 'POST':
       
        start = time.time()
        with counter.get_lock():
            counter.value += 1
            out = str(counter.value)
        res = request.data
        image = base64.b64decode(res)
        INPUT_IMAGE_DIR='.tree/'+out+'/input_tree'
        OUTPUT_IMAGE_DIR='.tree/'+out+'/output_tree'
        if not os.path.exists(INPUT_IMAGE_DIR):
            os.makedirs(INPUT_IMAGE_DIR)
        if not os.path.exists(OUTPUT_IMAGE_DIR):
            os.makedirs(OUTPUT_IMAGE_DIR)    
        image = Image.open(io.BytesIO(image))
        image = image.resize((512,512))
        image.convert('RGB').save(INPUT_IMAGE_DIR+'/'+str('1')+'.jpg', optimize=True, quality=100)
       

        data =requests.post("http://localhost:3000/predicttree", files={'file': open(INPUT_IMAGE_DIR+'/'+str('1')+'.jpg', 'rb')})

        return jsonify(json.loads(data.text))
@cross_origin()
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        start = time.time()
        with counter.get_lock():
            counter.value += 1
            out = str(counter.value)
        res = request.data
        image = base64.b64decode(res)
        INPUT_IMAGE_DIR='./app/'+out+'/INPUT_IMAGE'
        if not os.path.exists(INPUT_IMAGE_DIR):
            os.makedirs(INPUT_IMAGE_DIR)
        image = Image.open(io.BytesIO(image))
        image = image.resize((4000,3000))
        image.convert('RGB').save(INPUT_IMAGE_DIR+'/'+str('1')+'.jpg', optimize=True, quality=100)
        prediction = get_prediction(out)
        encoded_img = get_response_image('./app/'+out+'/OUTPUT_IMAGE/pred.jpg')
        end = time.time()
        data = {'prediction': prediction,'time':"{:.2f}".format(end-start)+' secs','ImageBytes':encoded_img}
        return jsonify(data)
       
@cross_origin()
@app.route('/predictmango', methods=['POST'])
def predictmango():
    if request.method == 'POST':
        start = time.time()
        with counter.get_lock():
            counter.value += 1
            out = str(counter.value)
        res = request.data
        image = base64.b64decode(res)
        INPUT_IMAGE_DIR='./mango/'+out+'/INPUT_IMAGE'
        if not os.path.exists(INPUT_IMAGE_DIR):
            os.makedirs(INPUT_IMAGE_DIR)
        image = Image.open(io.BytesIO(image))
        image = image.resize((4000,3000))
        image.convert('RGB').save(INPUT_IMAGE_DIR+'/'+str('1')+'.jpg', optimize=True, quality=100)
        prediction = get_prediction_mango(out)
        encoded_img = get_response_image('./mango/'+out+'/OUTPUT_IMAGE/pred.jpg')
        end = time.time()
        data = {'prediction': prediction,'time':"{:.2f}".format(end-start)+' secs','ImageBytes':encoded_img}
        return jsonify(data)
       


if __name__ == "__main__":
  app.run()