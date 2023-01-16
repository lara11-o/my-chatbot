from nltk.stem.lancaster import LancasterStemmer
import json
import numpy as np
import pickle
import logging
from urllib.request import urlretrieve
from PIL import Image
import random
from nltk.stem import WordNetLemmatizer
import streamlit as st
from streamlit_chat import message as st_message
from keras.models import load_model
import nltk
nltk.download('punkt')
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
# from transformers import BlenderbotTokenizer
# from transformers import BlenderbotForConditionalGeneration
# from working_chatbot import predict_class, get_response

# import pickle


class chat:

    def __init__(self, model_path):
        self.model = load_model(model_path)
        with open("words.pkl", 'rb') as handle:
            self.words = pickle.load(handle)
        with open('labels.pkl', 'rb') as handle:
            self.labels = pickle.load(handle)

    def clean_up_sentences(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [stemmer.stem(word.lower())
                          for word in sentence_words]
        return sentence_words

    def bagw(self, sentence):
        sentence_words = self.clean_up_sentences(sentence)
        bag = [0]*len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bagw(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res)
                   if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.labels[r[0]],
                                'probability': str(r[1])})
        return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            # break
    return result


with open("intents.json") as f:
    data = json.load(f)

    def main():
        model_path = 'chatbotmodel.h5'
        model = chat(model_path)
        st.write("Welcome to ChatBot!")

        widget_count = 0
        message = st.text_input("You: ", key="message"+str(widget_count))
        widget_count += 1
        if st.button('Send'):
                res = model.predict_class(message)
                result = get_response(res, data)
                st.success("BOT: " + result)
                #st.text_input(f"Bot : {result}")

        if message.lower() == "bye":
            #break
            st.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
