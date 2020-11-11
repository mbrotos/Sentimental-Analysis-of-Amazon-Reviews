import numpy as np
import pandas as pd
import random
import json

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer

def defaultFit(classifer, X_train, Y_train, X_test):
    classifer.fit(X_train, Y_train)
    predict = classifer.predict(X_test)
    return (classifer, predict)

def denseFit(classifer, X_train, Y_train, X_test):
    classifer.fit(X_train.todense(), Y_train)
    predict = classifer.predict(X_test.todense())
    return (classifer, predict)