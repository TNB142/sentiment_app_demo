import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
# import numpy as np
# import pandas as pd
from sentence_transformers import SentenceTransformer
# import re
from preprocessing_data import *

app = Flask(__name__, template_folder='template')

our_model=pickle.load(open('models/predict_model/svm_tf_data_clean.pkl','rb'))
# model=SentenceTransformer('paraphrase-MiniLM-L6-v2')
tf_vectorizer=pickle.load(open('models/tf_vec/tf_vect2_v2.pkl','rb'))



@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json()['data']
    data_clean=clean_data(str(data))
    data_encode=tf_vectorizer.transform([data_clean])
    output=our_model.predict(data_encode.reshape(1,-1)) if data_not_null(data_clean) else [6]
    name_class=class_name(int(output[0]))
    return jsonify({
        "0_text_before_clean":data,
        "1_after_clean":data_clean,
        "2_class_predict":name_class})

@app.route('/predict',methods=['POST'])
def predict():
    data= [x for x in request.form.values()]
    data_clean=clean_data(str(data[0]))
    data_encode=tf_vectorizer.transform([data_clean])
    # data_encode=count_vectorizer.transform(data)
    output=our_model.predict(data_encode.reshape(1,-1)) if data_not_null(data_clean) else [6]
    # print(int(output[0]))
    name_class=class_name(int(output[0]))
    return render_template("index2.html",
                           prediction_text="Your emotion right now is {}".format(name_class),
                           text_input=str(data[0]))


if __name__=="__main__":
    app.run(debug=False)




