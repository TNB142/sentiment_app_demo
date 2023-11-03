import numpy as np
import pandas as pd
import re
from deep_translator import GoogleTranslator
# import contextualSpellCheck
# import spacy

# nlp = spacy.load('en_core_web_sm')
# contextualSpellCheck.add_to_pipe(nlp)

#Needed
# def translate_to_En(sentence):
#     translated = GoogleTranslator(source='auto', target='en').translate(sentence)
#     return translated
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

#General
def clean_data(sentence):
    # sentence = MyMemoryTranslator(source='vi-VN', target='en-US').translate(sentence)
    sentence = GoogleTranslator(source='auto', target='en').translate(sentence)
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', sentence) #clean url
    sentence = re.sub('@[^\s]+','', sentence) #clean user
    sentence = sentence.lower() #low text
    # doc=nlp(sentence)
    # sentence=doc._.outcome_spellCheck if doc._.performed_spellCheck else sentence
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
