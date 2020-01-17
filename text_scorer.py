import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import os
import gensim
import time
import re
import difflib

# data = """The cat and her kittens
# They put on their mittens,
# To eat a Christmas pie.
# The poor little kittens
# They lost their mittens,
# And then they began to cry.
# O mother dear, we sadly fear
# We cannot go to-day,
# For we have lost our mittens."
# "If it be so, ye shall not go,
# For ye are naughty kittens."""

data = """"""

tokenizer = Tokenizer()


def pre_process():
    global data
    corpus_dir = "./txt"
    max_books = 1
    for txt_file in os.listdir(corpus_dir)[:max_books]:
        print(os.path.join(corpus_dir, txt_file))
        f = open(os.path.join(corpus_dir, txt_file), "r")
        strTxt = f.read()
        data = data + strTxt
    reg = re.compile('[A-Za-z\n ]')
    lstcrctrs = reg.findall(data)
    data = ''.join(lstcrctrs)
    print(data)


def dataset_preparation():
    corpus = data.lower().split('\n')
    corpus = [row for row in corpus if row != '' or len(row.split()) < 10]
    print(corpus)
    tokenizer.fit_on_texts(corpus)
    word_index = tokenizer.word_index
    total_words = len(word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]  # [0] exists because to_sequences outputs [[token]]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
            print(n_gram_sequence)

    # Normalise input sequence to same length
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Get x_train, Y_train
    x_train, Y_train = input_sequences[:, :-1], input_sequences[:, -1]

    Y_train = to_categorical(Y_train, num_classes=total_words)

    return [x_train, Y_train, max_sequence_len, total_words]


def create_model(x, y, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    epochs = 20

    model = Sequential()

    model.add(Embedding(total_words, 10, input_length=input_len))

    model.add(LSTM(500, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(total_words, activation='softmax'))

    optimiser = keras.optimizers.Adam(lr=1e-3, decay=0.1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    model.fit(x, y, epochs=epochs)

    return model


def generate_text(text, model):
    _, _, max_sequence_len, _ = dataset_preparation()
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)

    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    return output_word


def score():
    x_train, Y_train, max_sequence_len, total_words = dataset_preparation()

    model = create_model(x_train, Y_train, max_sequence_len, total_words)
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    while True:
        sentence_to_score = input("Please input a sentence which you'd like to score.")
        lst_sentence = sentence_to_score.split()  # For improvement, use token

        scores = []
        for i in range(1, len(lst_sentence)):
            words_soFar = ''.join(lst_sentence[:i])
            try:
                exemplar = generate_text(words_soFar, model)
                exemplar = word2vec[exemplar]
                actual = lst_sentence[i]
                actual = word2vec[actual]
            except KeyError:
                continue

            sm = difflib.SequenceMatcher(None, exemplar, actual)
            scores.append(sm.ratio())
        score = sum(scores) / len(scores)
        print(f"The score give to this sentence is {score} out of {len(scores)}")

    model.save(f'language_model-{time.time()}.model')


if __name__ == "__main__":
    pre_process()
    score()

