from flaskblog import app
from flask_login import login_required

import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as k
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from flask import request
from flask import jsonify,make_response
from flask import Flask
import tensorflow as tf
import pandas as pd
import json
#from flask_ngrok import run_with_ngrok

#run_with_ngrok(app)

def get_model(name):
	global modelnearby
	#modelnearby=load_model(name)
	print("MODEL LOADED!")
	return load_model(name)

def preprocess_image(image,target_size):
	if image.mode!="RGB":
		image=image.convert("RGB")
	image=image.resize(target_size)
	image=img_to_array(image)
	image=np.expand_dims(image,axis=0)
	return image


print("LOADING KERAS NEARBY TOM MODEL!")
modelnearbytom=get_model("tomato.h5")
global graphnearbytom
tf_config = some_custom_config
sesstom = tf.Session(config=tf_config)
graphnearbytom = tf.get_default_graph()

@app.route('/predicttom', methods=["GET","POST"])
def predicttom():
	message=request.get_json(force=True)
	encoded=message['image']
	decoded=base64.b64decode(encoded)
	image=Image.open(io.BytesIO(decoded))
	processed_image=preprocess_image(image,target_size=(256,256))
	with graphnearbytom.as_default():
		set_session(sesstom)
		prediction=modelnearbytom.predict(processed_image)
		response={
		
		'c0':prediction[0][0],
		'c1':prediction[0][1],
		}
	return pd.Series(response).to_json(orient='values')


print("LOADING KERAS NEARBY PEP MODEL!")
modelnearbypep=get_model("pepper.h5")
global graphnearbypep
tf_config = some_custom_config
sesspep = tf.Session(config=tf_config)
graphnearbypep  = tf.get_default_graph()

@app.route('/predictpep', methods=["GET","POST"])
def predictpep():
	message=request.get_json(force=True)
	encoded=message['image']
	decoded=base64.b64decode(encoded)
	image=Image.open(io.BytesIO(decoded))
	processed_image=preprocess_image(image,target_size=(256,256))
	with graphnearbypep.as_default():
		set_session(sesspep)
		prediction=modelnearbypep.predict(processed_image)
		response={
		
		'c0':prediction[0][0],
		'c1':prediction[0][1],
		}
	return pd.Series(response).to_json(orient='values')


print("LOADING KERAS NEARBY POT MODEL!")
modelnearbypot=get_model("potato.h5")
global graphnearbypot
tf_config = some_custom_config
sesspot = tf.Session(config=tf_config)
graphnearbypot  = tf.get_default_graph()

@app.route('/predictpot', methods=["GET","POST"])
def predictpot():
	message=request.get_json(force=True)
	encoded=message['image']
	decoded=base64.b64decode(encoded)
	image=Image.open(io.BytesIO(decoded))
	processed_image=preprocess_image(image,target_size=(256,256))
	with graphnearbypot.as_default():
		set_session(sesspot)
		prediction=modelnearbypot.predict(processed_image)
		response={
		
		'c0':prediction[0][0],
		'c1':prediction[0][1],
		}
	return pd.Series(response).to_json(orient='values')


print("LOADING KERAS NEARBY GRP MODEL!")
modelnearbygrp=get_model("grape.h5")
global graphnearbygrp
tf_config = some_custom_config
sessgrp = tf.Session(config=tf_config)
graphnearbygrp  = tf.get_default_graph()

@app.route('/predictgrp', methods=["GET","POST"])
def predictgrp():
	message=request.get_json(force=True)
	encoded=message['image']
	decoded=base64.b64decode(encoded)
	image=Image.open(io.BytesIO(decoded))
	processed_image=preprocess_image(image,target_size=(256,256))
	with graphnearbygrp.as_default():
		set_session(sessgrp)
		prediction=modelnearbygrp.predict(processed_image)
		response={
		
		'c0':prediction[0][0],
		'c1':prediction[0][1],
		}
	return pd.Series(response).to_json(orient='values')


print("LOADING KERAS FARBY MODEL!")
modelfarby=get_model("farbyre3.h5")
global graphfarby
graphfarby  = tf.get_default_graph()

	
@app.route('/predictfarby', methods=["GET","POST"])
def predictfarby():
	message=request.get_json(force=True)
	encoded=message['image']
	encoded=encoded+str("==")
	decoded=base64.b64decode(encoded)
	image=Image.open(io.BytesIO(decoded))
	processed_image=preprocess_image(image,target_size=(256,256))
	with graphfarby.as_default():
		prediction=modelfarby.predict(processed_image)
	response={
		
		'd1':prediction[0][0],
		'd2':prediction[0][1],
		'd3':prediction[0][2],
		'd4':prediction[0][3],
		'd5':prediction[0][4],
					}
	return pd.Series(response).to_json(orient='values')



if __name__ == '__main__':
    app.run()