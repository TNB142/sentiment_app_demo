# Sentiment Prediction Demo Application
Sentiment analysis has emerged as a crucial and captivating machine learning problem in recent years. Numerous methods exist for predicting people's emotions, whether derived from text, facial expressions, or gestures. In this application, a machine learning model was trained on a dataset of tweets and seamlessly integrated into an app. The backend was constructed using Flask, while the frontend was developed with HTML/CSS.

In detail, an SVM model was trained using the mix_CARER dataset, which is a combination of the CARER dataset and a tweet dataset. The mix_CARER dataset comprises 30,000 texts and labels. Further information, including details on a preprocessing project and a training project, will be provided in the Bonus Section.

Despite the model being trained on an English dataset, the application supports the deep-translator package, enabling the translation of other languages into English. However, it's important to note that using this translator may potentially reduce the accuracy of predictions.
## Setup Environment
Git clone

```
git clone https://github.com/TNB142/sentiment_test_app.git
```

Create conda environment

```
conda create -p venv python==3.10.6 -y
```

Download necessary library

```
cd sentiment_test_app
```

```
pip instal -r requirements.txt
```
## Features
### Home Page Section
<img src="https://hackmd.io/_uploads/rk2e_iPEa.png" alt="Alt text" title="Home Page Screen">

### Predict Page Section
<img src="https://hackmd.io/_uploads/B1vwdswET.png" alt="Alt text" title="Predict Screen">

<img src="https://hackmd.io/_uploads/Hk8_usvNT.png" alt="Alt text" title="Predict Vietnamese Screen">

