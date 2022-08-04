import jsonify as jsonify
from keras.models import load_model
import pickle
import numpy as np
import nltk
from nltk import LancasterStemmer
import pandas as pd
import train_data

stemmer = LancasterStemmer()

model = load_model("chatbot_model.hdf5")

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('documents.pkl', 'rb') as f:
    documents = pickle.load(f)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))


def classify(sentence):
    ERROR_THRESHOLD = 0.25
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input']).to_numpy()
    results = model.predict([input_data])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list


# def get_prediction(sentence):
#         prediction = predictStringInput(chatbot_model,sentence)
#     return prediction
