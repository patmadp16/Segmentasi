from flask import Flask, redirect, url_for, render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
from operasiFile import hitungTotalFile

import pandas as pd
import numpy as np
import os
import uuid
import base64


from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


#import keras as keras
import keras._tf_keras.keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
#rom keras.models import load_model
#from keras.preprocessing import image


model = load_model('static/model/klasifikasi_model.h5')

model.make_predict_function()
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	prediction  = np.max(model.predict(i))
	confidence  = round((prediction * 100),2)
	if prediction == 1 :
		label = "Kanker"
	else:
		label = "Bukan Kanker"
	return label, confidence

BASE_DIR = os.getenv("BASE_DIR")

UPLOAD_FOLDER = "data_upload"
BASE_URL = os.getenv("SERVER_URL")

anggrek_class = ["non kanker", "kanker usus"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# route index
@app.route("/")
def index():
    return render_template("home.html", dr=BASE_URL)


@app.route('/klasifikasi')
def klasifikasi():
    return render_template('classify.html', mData=BASE_URL)


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		ap, confidence = predict_label(img_path)

	return render_template("classification.html", prediction = ap, img_path = img_path, confidence=confidence)


# jalankan server 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)