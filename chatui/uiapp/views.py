from django.shortcuts import render

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import numpy
import tflearn
import tensorflow
import random
import json
import pickle
# Create your views here.

model, labels, data = None, None, None


def home(request):
    init_variables()
    return render(request, 'home.html')


def reply_to_chat(request):
    inp = request.POST['query']
    results = model.predict([bag_of_words(inp, words)])[0]
    print('####################', results)
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    if results[results_index] > 0.2:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
    else:
        print('I dnt understand!')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def init_variables():
    global data
    global words, labels, training, output , model

    with open("intents.json") as file:
        data = json.load(file)

    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.load("model.tflearn")
