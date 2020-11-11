import numpy as np
import pandas as pd
import random
import json

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer

def bagOfWordsAll(data, token):
    cvector = CountVectorizer(stop_words='english', ngram_range=(1,1), tokenizer=token.tokenize, max_features= 500)
    text_t = cvector.fit_transform(data['Summary and Review'])
    return text_t

def tfidf(data, token):
    tfidf = TfidfVectorizer(stop_words='english', min_df=50, tokenizer=token.tokenize)
    text_t2 = tfidf.fit_transform(data['Summary and Review'] )
    return text_t2

# def bagOfWordsAll_NEG(data, token):
#     x = " . ".join(x.split('.')).split()
#     from nltk.sentiment.util import mark_negation
#     x = mark_negation(x)
#     x = " ".join(x)
#     return x