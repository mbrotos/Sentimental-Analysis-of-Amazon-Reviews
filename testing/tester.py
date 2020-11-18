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


import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer

import variations
import classifiers

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

classifers = [
    ((LinearSVC(max_iter=100000)),0,'SVM L1'),
    (MultinomialNB(), 0, 'MultinomialNB')
]


variationOptions = {
    0 : variations.tfidf,
    1 : variations.bow,
    2 : variations.bow_filter

}

classiferOptions = {
    0 : classifiers.defaultFit,
    1 : classifiers.denseFit
}

def testClassifers(data, vOption=3):
    text_t = variationOptions[vOption](data, token.tokenize) # Vectorizes data based on a variation option
    X_train, X_test, Y_train, Y_test = train_test_split(text_t, data['Rating'], test_size=0.25, random_state=5)

    print(str(variationOptions[vOption]))
    for classifer, cOption, name in classifers:
        classifer, predicted = classiferOptions[cOption](classifer, X_train, Y_train, X_test) #Fits the classifer given unique option
        accuracy_score = metrics.accuracy_score(predicted, Y_test)
        
        print(str(name+': {:04.2f}'.format(accuracy_score*100))+'%\n')
        print(metrics.confusion_matrix(Y_test, predicted))
        print('\n\n')

if __name__ == "__main__":
    data = pd.read_csv('merged.csv')
    for i in range(len(list(variationOptions.keys()))):
        testClassifers(data, vOption=i) 
