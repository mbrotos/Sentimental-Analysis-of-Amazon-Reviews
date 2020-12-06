import numpy as np
import pandas as pd
import random
import json
import re

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.util import mark_negation
"""
Helper Methods
"""


def readwords(filename):
    f = open(filename, encoding="ISO-8859-1")
    words = [line.rstrip() for line in f.readlines()]
    f.close()
    return words


def filterData(data):
    positive = readwords('positive-words.txt')
    negative = readwords('negative-words.txt')
    negative = negative[36:]
    positive = positive[36:]
    stemmer = PorterStemmer()
    negative = [stemmer.stem(w) for w in negative]
    positive = [stemmer.stem(w) for w in positive]
    negative = list(dict.fromkeys(negative))
    positive = list(dict.fromkeys(positive))

    ps = {k: v for v, k in enumerate(positive)}
    ns = {k: v for v, k in enumerate(negative)}

    sample = []
    for i in range(len(data)):
        if('\n' in data.iloc[i]['Summary and Review']):
            temp = ''.join(data.iloc[i]['Summary and Review'].split('\n'))
        else:
            temp = data.iloc[i]['Summary and Review']
        temp = " . ".join(temp.split('.')).split()
        sample.append(temp)
    #negation
    for i in range(len(sample)):
        mark_negation(sample[i], shallow=True)
    
    for i in range(len(sample)):
        temp = []
        for w in sample[i]:
            s = stemmer.stem(w)
            if('NEG' in w):
                temp.append('NEG')
            elif(s in ps or s in ns):
                temp.append(w)
        sample[i] = ' '.join(temp)
    data['Summary and Review'] = sample
    return data


def stemData(data):
    stm = []
    stemmer = PorterStemmer()
    for rev in data['Summary and Review']:
        c = re.findall('[a-zA-Z0-9]+', rev)
        c = [stemmer.stem(plural) for plural in c]
        stm.append(' '.join(c))
    data['Summary and Review'] = stm
    return data


def writeToFile(name, data):
    file1 = open(name, "w")
    file1.write(','.join(data))
    file1.close()
"""
Variations
"""


def tfidf(data, token, fileName = "tfidf.txt"):
    tfidf = TfidfVectorizer(stop_words='english', min_df=50, tokenizer=token)
    text_t2 = tfidf.fit_transform(data['Summary and Review'])
    writeToFile(fileName, tfidf.get_feature_names())
    return text_t2


def bow(data, token, fileName = "bow.txt"):
    cvector = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token)
    text_t = cvector.fit_transform(data['Summary and Review'])
    writeToFile(fileName, cvector.get_feature_names())
    return text_t


# def bow_filter(data, token):
#     data = filter(data)
#     cvector = CountVectorizer(stop_words='english', ngram_range=(1, 1))
#     text_t = cvector.fit_transform(data['Summary and Review'])
#     return text_t


# def bow_filter_freq(data, token):
#     data = filter(data)
#     return bow_freq(data, None)


# def bow_stem(data, token):
#     data = stemmer(data)
#     cvector = CountVectorizer(stop_words='english', ngram_range=(1, 1))
#     text_t = cvector.fit_transform(data['Summary and Review'])
#     return text_t


# def bow_stem_freq(data, token):
#     data = stemmer(data)
#     return bow_freq(data, None)


# def bow_stem_filter(data, token):
#     data = filter(data)
#     data = stemmer(data)
#     cvector = CountVectorizer(stop_words='english', ngram_range=(1, 1))
#     text_t = cvector.fit_transform(data['Summary and Review'])
#     return text_t


# def bow_stem_filter_freq(data, token):
#     data = filter(data)
#     data = stemmer(data)
#     return bow_freq(data, None)


def bow_freq(data, token):
    positive = readwords('positive-words.txt')
    negative = readwords('negative-words.txt')
    negative = negative[36:]
    positive = positive[36:]
    stemmer = PorterStemmer()
    negative = [stemmer.stem(w) for w in negative]
    positive = [stemmer.stem(w) for w in positive]
    negative = list(dict.fromkeys(negative))
    positive = list(dict.fromkeys(positive))

    ps = {k: v for v, k in enumerate(positive)}
    ns = {k: v for v, k in enumerate(negative)}

    sample = []
    for i in range(len(data)):
        if('\n' in data.iloc[i]['Summary and Review']):
            temp = ''.join(data.iloc[i]['Summary and Review'].split('\n'))
        else:
            temp = data.iloc[i]['Summary and Review']
        temp = " . ".join(temp.split('.')).split()
        sample.append(temp)
    #negation
    for i in range(len(sample)):
        mark_negation(sample[i], shallow=True)

    for i in range(len(sample)):
        temp = []
        for w in sample[i]:
            s = stemmer.stem(w)
            if('NEG' in w):
                temp.append('NEG')
            elif(s in ps or s in ns):
                temp.append(w)
        sample[i] = ' '.join(temp)
    ns['NEG'] = 0

    simple = []
    for r in range(len(sample)):
        tN = 0
        tP = 0
        l = len(data['Summary and Review'][r])
        row = sample[r].split()
        for c in range(len(row)):
            if(stemmer.stem(row[c]) in ps):
                tP += 1
            else:
                tN += 1
        simple.append([tP, tN, l])
    
    return simple


##############SAMPLE-RUN############
# data = pd.read_csv('merged.csv', index_col=0)
