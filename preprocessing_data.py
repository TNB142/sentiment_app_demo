import numpy as np
import pandas as pd
import re
from deep_translator import GoogleTranslator
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import SnowballStemmer, LancasterStemmer, PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
# import contextualSpellCheck
# import spacy

# nlp = spacy.load('en_core_web_sm')
# contextualSpellCheck.add_to_pipe(nlp)

#Needed
# def translate_to_En(sentence):
#     translated = GoogleTranslator(source='auto', target='en').translate(sentence)
#     return translated
#decontracted function
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"wont", "will not", phrase)
    phrase = re.sub(r"won\ t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"cant", "can not", phrase)
    phrase = re.sub(r"can\ t", "can not", phrase)

    phrase = re.sub(r"didt", "did not", phrase)
    phrase = re.sub(r"didnt", "did not", phrase)
    phrase = re.sub(r"didn\ t", "did not", phrase)

    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"don\ t", "do not", phrase)

    phrase = re.sub(r"doest", "does not", phrase)
    phrase = re.sub(r"doesnt", "does not", phrase)
    phrase = re.sub(r"doesn\ t", "does not", phrase)

    phrase = re.sub(r"isnt", "is not", phrase)
    phrase = re.sub(r"isn\ t", "is not", phrase)

    phrase = re.sub(r"arent", "are not", phrase)
    phrase = re.sub(r"aren\ t", "are not", phrase)

    phrase = re.sub(r"weve", "we have", phrase)
    phrase = re.sub(r"we ve", "we have", phrase)
    phrase = re.sub(r"we re", "we are", phrase)
    phrase = re.sub(r"i m","i am",phrase)
    phrase = re.sub(r"it s","it is",phrase)
    phrase = re.sub(r"he s","he is",phrase)
    phrase = re.sub(r"she s","she is",phrase)

    phrase = re.sub(r"i ve","i have",phrase)
    phrase = re.sub(r"they re","they are",phrase)
    phrase = re.sub(r"they ve","they have",phrase)


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
slang_abbrev_dict = {
    'AFAIK': 'As Far As I Know',
    'AFK': 'Away From Keyboard',
    'ASAP': 'As Soon As Possible',
    'ATK': 'At The Keyboard',
    'ATM': 'At The Moment',
    'A3': 'Anytime, Anywhere, Anyplace',
    'BAK': 'Back At Keyboard',
    'BBL': 'Be Back Later',
    'BBS': 'Be Back Soon',
    'BFN': 'Bye For Now',
    'B4N': 'Bye For Now',
    'BRB': 'Be Right Back',
    'BRT': 'Be Right There',
    'BTW': 'By The Way',
    'B4': 'Before',
    'B4N': 'Bye For Now',
    'CU': 'See You',
    'CUL8R': 'See You Later',
    'CYA': 'See You',
    'FAQ': 'Frequently Asked Questions',
    'FC': 'Fingers Crossed',
    'FWIW': 'For What It\'s Worth',
    'FYI': 'For Your Information',
    'GAL': 'Get A Life',
    'GG': 'Good Game',
    'GN': 'Good Night',
    'GMTA': 'Great Minds Think Alike',
    'GR8': 'Great!',
    'G9': 'Genius',
    'IC': 'I See',
    'ICQ': 'I Seek you',
    'ILU': 'I Love You',
    'IMHO': 'In My Humble Opinion',
    'IMO': 'In My Opinion',
    'IOW': 'In Other Words',
    'IRL': 'In Real Life',
    'KISS': 'Keep It Simple, Stupid',
    'LDR': 'Long Distance Relationship',
    'LMAO': 'Laugh My Ass Off',
    'LOL': 'Laughing Out Loud',
    'LTNS': 'Long Time No See',
    'L8R': 'Later',
    'MTE': 'My Thoughts Exactly',
    'M8': 'Mate',
    'NRN': 'No Reply Necessary',
    'OIC': 'Oh I See',
    'OMG': 'Oh My God',
    'PITA': 'Pain In The Ass',
    'PRT': 'Party',
    'PRW': 'Parents Are Watching',
    'QPSA?': 'Que Pasa?',
    'ROFL': 'Rolling On The Floor Laughing',
    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',
    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',
    'SK8': 'Skate',
    'STATS': 'Your sex and age',
    'ASL': 'Age, Sex, Location',
    'THX': 'Thank You',
    'TTFN': 'Ta-Ta For Now!',
    'TTYL': 'Talk To You Later',
    'U': 'You',
    'U2': 'You Too',
    'U4E': 'Yours For Ever',
    'WB': 'Welcome Back',
    'WTF': 'What The Fuck',
    'WTG': 'Way To Go!',
    'WUF': 'Where Are You From?',
    'W8': 'Wait',
    '7K': 'Sick:-D Laugher'
}
def check_unslang(sentence):
    """Converts text like "OMG" into "Oh my God"
    """
    words = word_tokenize(sentence)
    all_words=[]
    for word in words:
      if word.upper() in slang_abbrev_dict.keys():
          all_words.append(slang_abbrev_dict[word.upper()])
      else:
          all_words.append(word)
    return " ".join(all_words)

white_tokenizer = WhitespaceTokenizer() #word in space
snow_stemmer=SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer() #wlemmatizer word

def lemmatize_stemming(sentence):
    text = sentence
    # text=' '.join([snow_stemmer.stem(w) for w in word_tokenize(str(sentence))])
    text=' '.join([lemmatizer.lemmatize(w) for w in white_tokenizer.tokenize(str(text))])
    return text
def preprocess_stemmer(sentence):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', sentence['data_clean']) #clean url
    text = lemmatize_stemming(str(text))
    return text
#General
def clean_data(sentence):
    # sentence = MyMemoryTranslator(source='vi-VN', target='en-US').translate(sentence)
    sentence = GoogleTranslator(source='auto', target='en').translate(sentence)
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', sentence) #clean url
    sentence = re.sub('@[^\s]+','', sentence) #clean user
    sentence = sentence.lower() #low text
    sentence = check_unslang(str(sentence))
    # doc=nlp(sentence)
    # sentence=doc._.outcome_spellCheck if doc._.performed_spellCheck else sentence
    sentence = decontracted(sentence)
    # sentence = spell_check2(str(sentence))
    sentence = re.sub('&[^\s]+;', '', sentence) #xóa html bắt đầu bằng &
    sentence = re.sub('[^a-zA-Za-яА-Я1-9]+', ' ', sentence) #xóa tất cả các lại dấu kí hiệu
    sentence = re.sub(' +',' ', sentence) # xóa câu chữ nhấn nhiều space
    # sentence = lemmatize_stemming(str(sentence))
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
