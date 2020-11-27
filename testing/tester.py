import numpy as np
import pandas as pd
import random
import json

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer

import variations
import classifiers

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

classifers = [
    (LogisticRegression(max_iter=1000, random_state=1),0,'Logistic Regression')
]
"""
    , 
    (MultinomialNB(random_state=1), 0, 'MultinomialNB'),
    ((LinearSVC(random_state=1)),0,'Linear SVM')
"""

variationOptions = {
    0 : variations.tfidf,
    1 : variations.bow,
    2 : variations.bow_filter,
    3 : variations.bow_stem,
    4 : variations.tfidf_filter,
    5 : variations.tfidf_stem,
    6 : variations.bow_stem_filter,
    7 : variations.tfidf_stem_filter
}

classiferOptions = {
    0 : classifiers.defaultFit,
    1 : classifiers.denseFit
}
def getMislabelled(predicted, y_val):
    return [i for i in range(y_val) if predicted[i] != y_val[i]] # Indices of mislabelled samples

def testClassifers(data, vOption=3):
    dataCopy = data.copy()
    text_t = variationOptions[vOption](dataCopy, token.tokenize) # Vectorizes data based on a variation option
    X_train, X_test, y_train, y_test = train_test_split(text_t, dataCopy['Rating'], test_size=0.2, random_state=1) # Note test is never used!!
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    print(str(variationOptions[vOption]))
    for classifer, cOption, name in classifers:
        classifer, predicted = classiferOptions[cOption](classifer, X_train, y_train, X_val) #Fits the classifer given unique option
        accuracy_score = metrics.accuracy_score(predicted, y_val)
        
        print(str(name+': {:04.2f}'.format(accuracy_score*100))+'%\n')
        print(metrics.confusion_matrix(y_val, predicted))
        mislabelled = getMislabelled(y_val, predicted)
        print('\n\n')

if __name__ == "__main__":
    data = pd.read_csv('merged.csv')
    testClassifers(data, vOption=0) 
