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

def bagOfWordsAll_NEG(data, token):
    '''
    Negation code from eisa using positive-words.txt and negative-words.txt
    '''
    sample = []
    for i in range(len(data)):
        temp = ''.join(data.iloc[i]['Summary and Review'].split('\n')).split()
        temp = ''.join(data.iloc[i]['Summary and Review'].split('\n'))
        temp = " . ".join(temp.split('.')).split()
        sample.append(temp)

    x = []
    from nltk.sentiment.util import mark_negation
    for i in range(len(sample)):
        x.append(mark_negation(sample[i]))
    for i in range(len(x)):
        x[i] = list(value for value in x[i] if value != ".")

    def readwords(filename):
        f = open(filename, encoding = "ISO-8859-1")
        words = [ line.rstrip() for line in f.readlines()]
        f.close()
        return words

    positive = readwords('positive-words.txt')
    negative = readwords('negative-words.txt')
    negative = negative[36:]
    positive = positive[36:]
    count_neg_temp = 0
    count_neg = []
    x_copy = x.copy()
    for i in range(len(x_copy)):
        for j in range(len(x_copy[i])):
            if "NEG" in x_copy[i][j]:
                count_neg_temp+=1
        count_neg.append(count_neg_temp)
        count_neg_temp = 0

    def intersection(lst1, lst2): 
        return list(set(lst1) & set(lst2))

    simple = []
    counter = 0
    for doc in sample:
        tNeg = len(intersection(doc,negative))
        tPos = len(intersection(doc,positive))
        lenDoc = len(doc)
        simple.append([tPos,tNeg + count_neg[counter],lenDoc])
        counter += 1
    return simple

