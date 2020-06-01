from flask import Flask,render_template,request,send_file,send_from_directory
import numpy as np
import pandas as pd
import sklearn.metrics as m
from keras.utils.np_utils import to_categorical
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Activation,MaxPooling2D
from keras.preprocessing import image
from keras.models import load_model
import keras.backend.tensorflow_backend as tb

from skimage import transform
import argparse
from keras.applications.vgg16 import VGG16
from keras.models import Model
import tensorflow as tf
from PIL import Image
from matplotlib import cm
import io

new_width  = 224
new_height = 224

tb._SYMBOLIC_SCOPE.value = True

model = load_model('model-facemask.h5')

def processesing(arr):
  for i in arr:
    if(i[0]>i[1]):
      return 0
    else:
      return 1

app = Flask(__name__,static_folder='static',template_folder='templates')

@app.route('/')
def home():
      return render_template("index.html")

def percentage(u,pre):
  sum=u[0][0]+u[0][1]
  return 100*u[0][pre]/sum

@app.route('/predict',methods=['POST','GET'])
def predict():
  if request.method=='POST':
    img=request.files['ima'].read()
    img = Image.open(io.BytesIO(img))

    img = img.resize((new_width, new_height), Image.ANTIALIAS)

    image = np.expand_dims(img, axis=0)

    imgarray=image

    imgarray=image
    u=model.predict(imgarray)
    pre=processesing(u)
    
    perc=percentage(u,pre)
    
    if pre==0:
      
      response="Mask ON! You are Safe"
      return render_template("prediction.html",predict=response,percent=str(perc)+" %")

    if pre==1:
      
      response="Mask OFF! Please wear the Mask"
      return render_template("prediction.html",predict=response,percent=str(perc)+" %")

  if request.method=='GET':
    return render_template("index.html")

     
if __name__=='__main__':
    app.run(port=80,debug=False,threaded=False)
