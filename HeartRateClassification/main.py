from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

import torch
from PIL import Image
import albumentations as aug
#from efficientnet_pytorch import EfficientNet
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = torch.load('canc.pth')
model.eval()

@app.route("/")
def index():
	return render_template("index.html")

def model_predict(file, model):
    image = Image.open(file)
    image = np.array(image)
    transforms = aug.Compose([
            aug.Resize(224,224),
            aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225),max_pixel_value=255.0,always_apply=True),
            ])
    image = transforms(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor([image], dtype=torch.float)
    preds = model(image)
    preds = np.argmax(preds.detach())
    return preds

def save_image(url):
    response =request.get(url)
    if response.status_code == 200:
        with open("img.png",'wb') as f:
           f.write(response.content)


@app.route('/predict', methods=['POST'])
def upload():
    skin_lesion=request.get_json()
    image_url=skin_lesion['url']
    print(image_url)
    save_image(image_url)
    labs= ['MELANOMA', 'MELANOCYTIC NEVUS', 'BASAL CELL CARCINOMA', 'ACTINIC KERATOSIS', 'BENIGN KERATOSIS', 'DERMATOFIBROMA', 'VASCULAR LESION', 'SQUAMOUS CELL CARCINOMA']
    preds = model_predict(image_url, model)
    result = labs[preds]
    return jsonify({'result':result})

if __name__ == '__main__':
	app.run(host="127.0.0.1",port=8080,debug=True)

