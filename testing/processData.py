import pandas as pd
import random
import json

def process(file):
    rev = []
    with open('Software_5.json') as f:
        for line in f:
            rev.append(json.loads(line))

    #main dataframe
    data = pd.DataFrame(columns=['Summary and Review', 'Rating'])

    #adding data into dataframe 
    for i in range(len(rev)):
        if('overall' in rev[i] and 'reviewText' in rev[i] and 'summary' in rev[i]):
            data.loc[i] = [rev[i]['summary'] + ' ' + rev[i]['reviewText'], rev[i]['overall']]
    #function to turn ratings into two class
    def rate(x):
        if(x>3):
            return 1
        elif(x<3):
            return 0
        else:
            return random.randint(0,1)
    data['Rating'] = data['Rating'].apply(rate)

    return data

if __name__ == "__main__":
    data = process(file = "Software_5.json")
    data.to_csv('dataFrame.csv', index=False)