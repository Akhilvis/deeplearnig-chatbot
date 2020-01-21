from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.http.response import JsonResponse

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


@csrf_exempt
def reply_to_chat(request):
    print(7777, request.body)
    responses = ['I do not understand!']
    postdata = json.loads(request.body)
    inp = postdata['query']
    results = model.predict([bag_of_words(inp, words)])[0]
    print('####################', results)
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    print('tag>>>>>>>>>>>>>>>>>.', tag)
    if results[results_index] > 0.2:
        for tg in data["intents"]:
            print('tags....', tg['tag'])
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
        return JsonResponse({'response': random.choice(responses)})

    else:
        print('I dnt understand!')
        return JsonResponse({'response': responses})


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
    global words, labels, training, output
    global model

    with open(settings.BASE_DIR + "/static/data/intents.json") as file:
        data = json.load(file)

    with open(settings.BASE_DIR + "/.." + "/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.load(settings.BASE_DIR + "/../" + "model.tflearn")
