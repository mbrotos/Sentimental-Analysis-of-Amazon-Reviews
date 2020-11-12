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

"""
Helper Methods
"""
def filter(data):
    def readwords(filename):
        f = open(filename, encoding = "ISO-8859-1")
        words = [ line.rstrip() for line in f.readlines()]
        f.close()
        return words

    positive = readwords('positive-words.txt')
    negative = readwords('negative-words.txt')
    negative = negative[36:]
    positive = positive[36:]
    ps = {k: v for v, k in enumerate(positive)}
    ns = {k: v for v, k in enumerate(negative)}

    bog = []
    for rev in data['Summary and Review']:
        c = re.findall('[a-zA-Z0-9]+', rev)
        c = [word for word in c if (word in ps or word in ns)]
        bog.append(' '.join(c))
    data['Summary and Review'] = bog
    return data

def stemmer(data):
    stm = []
    stemmer = PorterStemmer()
    for rev in data['Summary and Review']:
        c = re.findall('[a-zA-Z0-9]+', rev)
        c = [stemmer.stem(plural) for plural in c]
        stm.append(' '.join(c))
    data['Summary and Review'] = stm
    return data

"""
Variations
"""

def tfidf(data, token):
    tfidf = TfidfVectorizer(stop_words='english', min_df=50, tokenizer=token)
    text_t2 = tfidf.fit_transform(data['Summary and Review'] )
    return text_t2

def bow(data, token):
    cvector = CountVectorizer(stop_words='english', ngram_range=(1,1), tokenizer=token)
    text_t = cvector.fit_transform(data['Summary and Review'])
    return text_t

def bow_filter(data, token):
    data = filter(data)
    cvector = CountVectorizer(stop_words='english', ngram_range=(1,1))
    text_t = cvector.fit_transform(data['Summary and Review'])
    return text_t

def bow_filter_freq(data, token):
    data = filter(data)
    return bow_freq(data, None)

def bow_stem(data, token):
    data = stemmer(data)
    cvector = CountVectorizer(stop_words='english', ngram_range=(1,1))
    text_t = cvector.fit_transform(data['Summary and Review'])
    return text_t

def bow_stem_freq(data, token):
    data = stemmer(data)
    return bow_freq(data, None)

def bow_stem_filter(data, token):
    data = filter(data)
    data = stemmer(data)
    cvector = CountVectorizer(stop_words='english', ngram_range=(1,1))
    text_t = cvector.fit_transform(data['Summary and Review'])
    return text_t

def bow_stem_filter_freq(data, token):
    data = filter(data)
    data = stemmer(data)
    return bow_freq(data, None)
    
def bow_freq(data, token):
    '''
    Appends positive negative word frequencies to each review
    '''
    mainData = bow(data, token)
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
    mainData = mainData.toarray()
    simple = simple
    final = np.concatenate((mainData, simple), axis=1)
    return final

