import json as js
import random
import pickle
import torch
import re

from training import Classification
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_words(token):
    token = token.lower()
    token = re.sub("[^a-z0-9]*", "", token)
    token = lemmatizer.lemmatize(token)

    return token


def tokenize_sentences(sent, token2id, max_length=9, UNK=1, PAD=0):
    tokens = [token2id.get(clean_words(token), UNK) for token in sent.split()]

    if len(tokens) < max_length:
        diff = max_length - len(tokens)
        tokens.extend([PAD] * diff)
    elif len(tokens) > max_length:
        tokens = tokens[:max_length]

    return tokens


with open('./intents.json', 'rb') as f:
    data = js.load(f)['intents']

with open('token2idx.pkl', 'rb') as f:
    token2idx = pickle.load(f)

with open('label2tag.pkl', 'rb') as f:
    label2tag = pickle.load(f)

num_embeddings = len(token2idx.keys())
num_labels = len(label2tag.keys())

model = Classification(num_embeddings, num_labels)
model.load_state_dict(torch.load('./model.pth'))

lemmatizer = WordNetLemmatizer()

while True:
    input_sent = input()
    tokenize_input = tokenize_sentences(input_sent, token2idx)
    input_tensor = torch.tensor(tokenize_input).view(len(tokenize_input), 1)

    outputs = model(input_tensor)

    label = torch.argmax(outputs)
    tag = label2tag[int(label)]

    for item in data:
        if item['tag'] == tag:
            break

    print(random.choice(item['responses']))
