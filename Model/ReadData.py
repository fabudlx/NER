from collections import Counter
from random import randint

import numpy as np

from Model import ModelsNN
from Model.ModelsNN import SENTENCE_LENGTH
from Model.WordEmbeddings.WordEmbedding import EMBEDDING_SIZE
from Model.WordEmbeddings.WordEmbedding import EmbeddingModel
from keras.preprocessing import sequence


def make_character_embeddings(word_list, size):
    character_embedding_list = np.array([np.array(list(reversed([ord(character) for character in word]))) for word in word_list])
    character_embedding_list = [element[:size] for element in character_embedding_list]


    padded_character_embedding_list = sequence.pad_sequences(character_embedding_list, maxlen=size, value=0, dtype='int32')
    return padded_character_embedding_list


def make_data_connl03(file = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_eng.train', w2v_class = None, pos_of_tag=3):

    print('reading data')
    raw_data = open(file, 'r').read()
    raw_data = [word.split() for word in raw_data.split('\n')]
    if w2v_class is None:
        w2v_class = EmbeddingModel()

    X_forward = []
    X_backward = []
    Y = []

    print('preprocessing data')
    unknown_words = set()
    previous_words_from_sentence = []
    for data_point in raw_data:
        if not data_point: #shows sentence end
            if previous_words_from_sentence:
                previous_words_from_sentence = list(reversed(previous_words_from_sentence))
            while previous_words_from_sentence:
                embedded_backwards_sentence, unknown = w2v_class.get_embedding_improved(previous_words_from_sentence)
                for word in unknown:
                    unknown_words.add(word)
                padded_backwards_sentence = pad_in_front(embedded_backwards_sentence)
                X_backward.append(padded_backwards_sentence)
                previous_words_from_sentence.pop()
        else:
            previous_words_from_sentence.append(data_point[0])

            embedded_forward_sentence, unknown = w2v_class.get_embedding_improved(previous_words_from_sentence)
            for word in unknown:
                unknown_words.add(word)
            padded_forward_sentence = pad_in_front(embedded_forward_sentence)
            X_forward.append(padded_forward_sentence)

            tag = data_point[pos_of_tag]
            if tag == 'O':
                Y.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif tag == 'I-ORG':
                Y.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif tag == 'B-ORG':
                Y.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
            elif tag == 'I-LOC':
                Y.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif tag == 'B-LOC':
                Y.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif tag == 'I-PER' :
                Y.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif tag == 'B-PER':
                Y.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif tag == 'I-MISC':
                Y.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif tag == 'B-MISC':
                Y.append([0, 0, 0, 0, 0, 0, 0, 0, 1])

    padded_character_embedding_list = make_character_embeddings([word_line[0] for word_line in raw_data if word_line],ModelsNN.CHAR_EMBEDDING_SIZE)

    X_forward = np.array(X_forward)
    X_backward = np.array(X_backward)
    Y = np.array([np.array(y) for y in Y])
    np.reshape(X_forward, (len(X_forward), ModelsNN.SENTENCE_LENGTH, EMBEDDING_SIZE))
    np.reshape(X_backward, (len(X_backward), ModelsNN.SENTENCE_LENGTH, EMBEDDING_SIZE))
    np.reshape(Y, (len(Y), ModelsNN.NO_OF_CATEGORIES))
    np.reshape(padded_character_embedding_list, (len(X_forward), ModelsNN.CHAR_EMBEDDING_SIZE))

    print('data read and preprocessed')

    return X_forward, X_backward, Y,padded_character_embedding_list, unknown_words

def make_data_germEval(file =r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\germeval\NER-de-train.tsv', w2v_class = None, pos_of_tag=2):

    print('reading data')
    raw_data = open(file, 'r', encoding="utf8").read()
    raw_data = [word.split() for word in raw_data.split('\n')]
    if w2v_class is None:
        w2v_class = EmbeddingModel()

    X_forward = []
    X_backward = []
    Y = []

    print('preprocessing data')
    unknown_words = set()
    previous_words_from_sentence = []
    for data_point in raw_data:
        if not data_point: #shows sentence end
            if previous_words_from_sentence:
                previous_words_from_sentence = list(reversed(previous_words_from_sentence))
            while previous_words_from_sentence:
                embedded_backwards_sentence, unknown = w2v_class.get_embedding_improved(previous_words_from_sentence)
                for word in unknown:
                    unknown_words.add(word)

                padded_backwards_sentence = pad_in_front(embedded_backwards_sentence)

                X_backward.append(padded_backwards_sentence)
                previous_words_from_sentence.pop()
        else:
            if data_point[0] != '#':
                previous_words_from_sentence.append(data_point[1])

                embedded_forward_sentence, unknown = w2v_class.get_embedding_improved(previous_words_from_sentence)
                for word in unknown:
                    unknown_words.add(word)

                padded_forward_sentence = pad_in_front(embedded_forward_sentence)
                X_forward.append(padded_forward_sentence)

                tag = data_point[pos_of_tag]
                if tag.startswith('O') :
                    Y.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
                elif tag.startswith('I-ORG'):
                    Y.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
                elif tag.startswith('B-ORG'):
                    Y.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
                elif tag.startswith('I-LOC'):
                    Y.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
                elif tag.startswith('B-LOC'):
                    Y.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
                elif tag.startswith('I-PER'):
                    Y.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
                elif tag.startswith('B-PER'):
                    Y.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
                elif tag.startswith('I-OTH'):
                    Y.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
                elif tag.startswith('B-OTH'):
                    Y.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
                else:
                    print('not working tag ',tag)


    padded_character_embedding_list = make_character_embeddings([word_line[1] for word_line in raw_data if (word_line and word_line[0] != '#')],ModelsNN.CHAR_EMBEDDING_SIZE)

    X_forward = np.array(X_forward)
    X_backward = np.array(X_backward)
    Y = np.array([np.array(y) for y in Y])

    np.reshape(X_forward, (len(X_forward), ModelsNN.SENTENCE_LENGTH, EMBEDDING_SIZE))
    np.reshape(X_backward, (len(X_forward), ModelsNN.SENTENCE_LENGTH, EMBEDDING_SIZE))
    np.reshape(Y, (len(X_forward), ModelsNN.NO_OF_CATEGORIES))
    print(padded_character_embedding_list.shape)
    np.reshape(padded_character_embedding_list, (len(X_forward), ModelsNN.CHAR_EMBEDDING_SIZE))

    print('data read and preprocessed')

    return X_forward, X_backward, Y, padded_character_embedding_list, unknown_words


def pad_in_front(embedded_sentence):
    if len(embedded_sentence) >= SENTENCE_LENGTH:
        return embedded_sentence[len(embedded_sentence)-SENTENCE_LENGTH:]
    else:
        return list((np.zeros((SENTENCE_LENGTH-len(embedded_sentence),EMBEDDING_SIZE), dtype=np.float32))) + list(embedded_sentence)

COUNTER = 0


def get_data_regulized(X_forward_train, X_backward_train, Y_train, bach_size = None):
    global COUNTER
    COUNTER += randint(0, 5000)

    X_forward_batch = []
    X_backward_batch = []
    Y_batch_true = []
    category_counter = {i: 0 for i in range(ModelsNN.NO_OF_CATEGORIES)}
    Y_number = [np.argmax(value) for value in Y_train]
    if bach_size is None:
        category_total =  Counter(Y_number)
        max_elements_per_category = category_total[min(category_total)]/2
        bach_size = max_elements_per_category*ModelsNN.NO_OF_CATEGORIES
    else:
        max_elements_per_category = int(bach_size / ModelsNN.NO_OF_CATEGORIES)

    while len(Y_batch_true) < bach_size :

        if COUNTER >= len(Y_number):
            COUNTER = 0

        if category_counter[Y_number[COUNTER]] < max_elements_per_category:
            X_forward_batch.append(X_forward_train[COUNTER])
            X_backward_batch.append(X_backward_train[COUNTER])
            Y_batch_true.append(Y_train[COUNTER])
            category_counter[Y_number[COUNTER]] += 1
            # print('size -->',len(Y_batch_true),'+++++++++',category_counter)
        COUNTER += 1
    # print(category_counter)
    X_forward_batch = np.array(X_forward_batch)
    X_backward_batch = np.array(X_backward_batch)
    Y_batch_true = np.array(Y_batch_true)
    np.reshape(X_forward_batch, (len(X_forward_batch), ModelsNN.SENTENCE_LENGTH, EMBEDDING_SIZE))
    np.reshape(X_backward_batch, (len(X_backward_batch), ModelsNN.SENTENCE_LENGTH, EMBEDDING_SIZE))
    np.reshape(Y_batch_true, (len(Y_batch_true),ModelsNN.NO_OF_CATEGORIES))


    return X_forward_batch, X_backward_batch, Y_batch_true

