import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import json
import numpy
import random

# import  tensorflow
# import tflearn

with open('intents.json') as file:
    data = json.load(file)
words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:

    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))
print(docs_x)
print(docs_y)

trainig = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_raw =out_empty[:]
    output_raw[labels.index(docs_y[x])] = 1

    trainig.append(bag)
    output.append(output_raw)

print(trainig)
print(output)