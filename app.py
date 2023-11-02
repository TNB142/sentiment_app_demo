import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re

app = Flask(__name__, template_folder='template')
our_model=pickle.load(open('logistic_embedding_transformer_data_clean.pkl','rb'))
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
def clean_data(sentence):
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', sentence) #clean url
    sentence = re.sub('@[^\s]+','', sentence) #clean user
    sentence = sentence.lower() #low text
    sentence = decontracted(sentence)
    # sentence = spell_check2(str(sentence))
    sentence = re.sub('&[^\s]+;', '', sentence) #xóa html bắt đầu bằng &
    sentence = re.sub('[^a-zA-Za-яА-Я1-9]+', ' ', sentence) #xóa tất cả các lại dấu kí hiệu
    sentence = re.sub(' +',' ', sentence) # xóa câu chữ nhấn nhiều space
    return sentence
def class_name(res):
    if res == 0:
        res="sadness"
        return res

    if res == 1:
        res="joy"
        return res

    if res == 2:
        res="love"
        return res

    if res == 3:
        res="anger"
        return res

    if res == 4:
        res="fear"
        return res
    if res == 5:
        res="surprise"
        return res
    return res

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json()['data']
    # print(data)
    data=clean_data(str(data))
    data_encode=model.encode(data)
    # print(data_encode.reshape(1,-1))
    output=our_model.predict(data_encode.reshape(1,-1))
    # print(int(output[0]))
    name_class=class_name(int(output[0]))
    return jsonify(name_class)

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




