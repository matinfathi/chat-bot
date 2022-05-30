import json as js
import pickle
import torch
import re

from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Arguments
BATCH_SIZE = 8
EMBEDDING_DIM = 16
LR = 0.01
EPOCHS = 50


def clean_words(token):
    token = token.lower()
    token = re.sub("[^a-z0-9]*", "", token)
    token = lemmatizer.lemmatize(token)

    return token


def tokenize_sentences(sent, max_length, token2id, UNK=1, PAD=0):
    tokens = [token2id.get(clean_words(token), UNK) for token in sent.split()]

    if len(tokens) < max_length:
        diff = max_length - len(tokens)
        tokens.extend([PAD] * diff)
    elif len(tokens) > max_length:
        tokens = tokens[:max_length]

    return tokens


class Classification(torch.nn.Module):
    def __init__(self, num_embeddings, num_labels):
        super(Classification, self).__init__()
        self.embedd = torch.nn.Embedding(num_embeddings, 16, padding_idx=0)
        self.lstm = torch.nn.LSTM(16, 16)
        self.fc1 = torch.nn.Linear(16, num_labels)

    def forward(self, x):
        x = self.embedd(x)
        output, (hidden, cell) = self.lstm(x)
        x = hidden.view(hidden.shape[1], -1)
        x = torch.nn.functional.log_softmax(self.fc1(x), dim=1)

        return x


if __name__ == '__main__':
    with open('./intents.json', 'r') as f:
        data = js.load(f)['intents']

    tag2label = {item['tag'].lower(): idx for idx, item in enumerate(data)}
    label2tag = {v: k for (k, v) in tag2label.items()}

    lemmatizer = WordNetLemmatizer()

    token2idx = {'<PAD>': 0, '<UNK>': 1}
    raw_dataset, dataset = [], []
    counter = 2

    max_len = max([len(sent) for item in data for sent in item])

    for item in data:
        lablel = item['tag'].lower()

        for pattern in item['patterns']:
            sentence = ''

            for word in pattern.split():
                clean_word = clean_words(word)
                sentence += clean_word + ' '

                if clean_word not in token2idx.keys():
                    token2idx[clean_word] = counter
                    counter += 1

            raw_dataset.append(tuple([sentence.strip(), lablel]))

    for sentence, tag in raw_dataset:
        token_list = tokenize_sentences(sentence, max_len, token2idx)
        label = tag2label[tag]
        dataset.append(tuple([token_list, label]))

    num_embeddings = len(token2idx.keys())
    num_labels = len(tag2label.keys())

    idx2token = {v: k for k, v in token2idx.items()}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Classification(num_embeddings, num_labels)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    accuaracies, losses = [], []

    total_step = len(dataloader)
    for epoch in range(EPOCHS):
        correct, total = 0, 0
        loss = 0

        for i, (features, labels) in enumerate(dataloader):
            features = torch.stack(features, dim=0)
            labels = labels

            outputs = model(features)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (torch.argmax(outputs, dim=1) == labels).float().sum()
            total += labels.size(0)

        accuracy = (100 * correct) / total
        accuaracies.append(accuracy)
        losses.append(loss)

        # print the information
        if (epoch + 1) % 5 == 0:
            print(
                f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_step}], Loss: {round(loss.item(), 4)}'
                f', Accuracy: {accuracy} '
            )

    torch.save(model.state_dict(), './model.pth')
    with open('token2idx.pkl', 'wb') as f:
        pickle.dump(token2idx, f)
    with open('label2tag.pkl', 'wb') as f:
        pickle.dump(label2tag, f)

