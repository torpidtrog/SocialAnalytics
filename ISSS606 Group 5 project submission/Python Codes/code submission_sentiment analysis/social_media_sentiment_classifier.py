from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

import numpy as np
import pandas as pd


# Preprocess text (username and link placeholders) -- no necessary for sentiment analysis
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# load pretrained model from https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
# tokenizer and sentiment classifier
def initialize_model():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, config, model


# getting sentiment score of the text

def sentiment_score(data, column_name='text'):
    """
    :param data: dataframe that contains the social media text
    :param column_name: column name given to the text field
    :return: dataframe with normalized scores of positive, negative, and neutral and the final sentiment based on
             highest score
    """
    text_data = data[column_name]
    # initialize list to capture scores and final sentiment
    neg, neu, pos, sentiment = [], [], [], []
    # initialize model
    tokenizer, config, model = initialize_model()
    # initialize counter
    counter = 0
    # loop through each
    for text in data[column_name]:
        text = preprocess(text)  # preprocess text
        encoded_input = tokenizer(text, return_tensors='pt')  # tokenize text
        output = model(**encoded_input)  # run the model and generate scores
        scores = output[0][0].detach().numpy()  # return score in numpy format
        scores = softmax(scores)  # compute exponential of each sentiment group and divide by the sum of exponential
        max_score = np.argmax(scores)  # return the index of the max score
        # save score data in the lists prepared
        neg.append(scores[0])
        neu.append(scores[1])
        pos.append(scores[2])
        sentiment.append(config.id2label[max_score])
        counter += 1
        # for
        if counter % 1000 == 0:
            print('Number of rows processed: ', counter)
        else:
            pass
    # save columns of the results
    data['positive_score'] = pos
    data['negative_score'] = neg
    data['neutral_score'] = neu
    data['sentiment'] = sentiment

    print('Number of rows processed: ', counter)

    return data
