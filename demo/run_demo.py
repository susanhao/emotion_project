import csv
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns 

import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from model_func import *
from video_cam import *
from flask import Flask, render_template, Response


#i couldn't upload the models to github.  So please change this to where the model is
model_dir = '../face_models/'
face_model = Face_Model(model_dir, 'vggFace_finetune_val_loss_FER.json', 'vggFace_finetune_val_loss_1.105_63.1_FER.h5')

app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')

def gen(camera):
	while True:
		frame = camera.get_frame(face_model)
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
	return Response(gen(VideoCamera()),
					mimetype='multipart/x-mixed-replace; boundary=frame')

# def graph_feed():
# 	graph_data = 

if __name__ == '__main__':
	app.run(host='localhost',port = 2000, debug=True)