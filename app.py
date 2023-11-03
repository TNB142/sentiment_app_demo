import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
# import numpy as np
# import pandas as pd
from sentence_transformers import SentenceTransformer
# import re
from preprocessing_data import *
from deep_translator import GoogleTranslator


app = Flask(__name__, template_folder='template')

our_model=pickle.load(open('models/predict_model/logistic_embedding_transformer_data_clean.pkl','rb'))
model=SentenceTransformer('paraphrase-MiniLM-L6-v2')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json()['data']
    # print(data)
    # data = GoogleTranslator(source='auto', target='en').translate(data)
    data=clean_data(str(data))
    data_encode=model.encode(data)
    # print(data_encode.reshape(1,-1))
    output=our_model.predict(data_encode.reshape(1,-1))
    # print(int(output[0]))
    name_class=class_name(int(output[0]))
    return jsonify(data,name_class)

@app.route('/predict',methods=['POST'])
def predict():
    data= [x for x in request.form.values()]
    data=clean_data(str(data[0]))
    data_encode=model.encode(data)
    output=our_model.predict(data_encode.reshape(1,-1))
    # print(int(output[0]))
    name_class=class_name(int(output[0]))
    return render_template("index.html",prediction_text="Your emotion right now is {}".format(name_class))


if __name__=="__main__":
    app.run(debug=True)




