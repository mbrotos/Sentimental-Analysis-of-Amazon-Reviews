import json
from sklearn.naive_bayes import GaussianNB
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix

class Review(object):
    """
        A single review data type.
    """
    def __init__(self, overall, reviewText):
        self.overall = overall
        self.reviewText = reviewText

def as_review(dct):
    """
        Note not all reviews contain 'reviewText' or 'summary'
    """
    if dct.get('overall') == None or dct.get('reviewText')==None: #Checks for data
        return None
    return Review(dct.get('overall'), dct.get('reviewText'))

def make_Dictionary(train):
    all_words = []
    for review in train:
        words = review.reviewText.split()
        all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary

def extract_features(test, dictionary):
    features_matrix = np.zeros((len(test),3000))
    docID = 0
    for reivew in test:
        words = test.reviewText.split()
        for word in words:
            wordID = 0
            for i,d in enumerate(dictionary):
                if d[0] == word:
                    wordID = i
                    features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1
    return features_matrix

def main(jsonFile = "Software_5.json"):
    reviews = [] # list of review objects
    with open(jsonFile, 'r') as file:
        for line in file:
            obj = as_review(json.loads(line))
            if obj != None:
                reviews.append(obj)
            else: 
                continue
    
    train = reviews[:int(len(reviews)*.70)] # First 70% of data for training
    test = reviews[int(len(reviews)*.70):] # Last 30% for testing

    dictionary = make_Dictionary(train)
    train_labels = np.zeros(int(len(reviews)*.70))
    
if __name__ == "__main__":
    main()