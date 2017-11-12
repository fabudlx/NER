from keras.layers import LSTM, Conv1D, Flatten, Dropout, Dense, Input, concatenate, Embedding, Reshape
from keras.models import Model
from keras.optimizers import RMSprop
import tensorflow as tf

from keras_contrib.layers.crf import CRF
from Model.WordEmbeddings.WordEmbedding import EMBEDDING_SIZE

SENTENCE_LENGTH = 15
NO_OF_CATEGORIES = 9
CHAR_EMBEDDING_SIZE = 50


def create_mlp_model():
    print('creating new MLP model')

    forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    forward_dense_layer = Conv1D(filters=300, kernel_size=SENTENCE_LENGTH, activation='relu')(forward_input)
    forward_dense_layer_flat = Flatten()(forward_dense_layer)

    backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    backward_dense_layer = Conv1D(filters=300, kernel_size=SENTENCE_LENGTH, activation="relu")(backward_input)
    backward_dense_layer_flat = Flatten()(backward_dense_layer)

    merge_one = concatenate([forward_dense_layer_flat, backward_dense_layer_flat])
    dropout = Dropout(0.6)(merge_one)
    dense = Dense(300, activation='relu')(dropout)
    dropout2 = Dropout(0.6)(dense)
    dense2 = Dense(100, activation='relu')(dropout2)
    predictions = Dense(NO_OF_CATEGORIES, activation='softmax')(dense2)

    model = Model(inputs=[forward_input, backward_input], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('MLP model created and compiled')
    return model


def create_CNN_model():
    print('creating new CNN model')

    forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    forward_cnn_layer1 = Conv1D(filters=50, kernel_size=(2), activation='relu',padding="same")(forward_input)
    forward_cnn_layer2 = Conv1D(filters=50, kernel_size=(5), activation='relu', padding="same")(forward_input)

    backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    backward_cnn_layer1 = Conv1D(filters=50, kernel_size=(2), activation='relu', padding="same")(backward_input)
    backward_cnn_layer2 = Conv1D(filters=50,kernel_size=(5), activation='relu', padding="same")(backward_input)

    merge1 = concatenate([forward_cnn_layer1, backward_cnn_layer1])
    dropout1 = Dropout(0.6)(merge1)
    flatten1 = Flatten()(dropout1)
    dense1 = Dense(200, activation='relu')(flatten1)

    merge2 = concatenate([forward_cnn_layer2,backward_cnn_layer2])
    dropout2 = Dropout(0.6)(merge2)
    flatten2 = Flatten()(dropout2)
    dense2 = Dense(200, activation='relu')(flatten2)

    merge_complete = concatenate([dense1,dense2])
    dropout_complete = Dropout(0.6)(merge_complete)
    dense_complete = Dense(100, activation='relu')(dropout_complete)

    predictions = Dense(NO_OF_CATEGORIES, activation='softmax')(dense_complete)

    model = Model(inputs=[forward_input, backward_input], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('CNN model created and compiled')
    return model



def create_lstm_model():
    print('creating new LSTM model')

    forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    forward_lstm_layer = LSTM(300, activation='relu')(forward_input)

    backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    backward_lstm_layer = LSTM(300, activation='relu')(backward_input)

    merge_one = concatenate([forward_lstm_layer, backward_lstm_layer])
    dropout = Dropout(0.6)(merge_one)
    dense = Dense(300, activation='relu')(dropout)
    dropout2 = Dropout(0.6)(dense)
    dense2 = Dense(100, activation='relu')(dropout2)
    predictions = Dense(NO_OF_CATEGORIES, activation='softmax')(dense2)

    model = Model(inputs=[forward_input, backward_input], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('LSTM model created and compiled')
    return model


def create_lstm_crf_model():
    print('creating new LSTM-CRF model')

    forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    forward_lstm_layer = LSTM(300, activation='relu')(forward_input)

    backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    backward_lstm_layer = LSTM(300, activation='relu')(backward_input)

    merge_one = concatenate([forward_lstm_layer, backward_lstm_layer])
    dropout = Dropout(0.6)(merge_one)
    dense = Dense(300, activation='relu')(dropout)
    dropout2 = Dropout(0.6)(dense)
    dense2 = Dense(100, activation='relu')(dropout2)
    dense3 = Dense(NO_OF_CATEGORIES, activation='relu')(dense2)
    reshape = Reshape((1,9))(dense3)

    crf = CRF(NO_OF_CATEGORIES, sparse_target=False)
    pred = crf(reshape)
    model = Model(inputs=[forward_input, backward_input], outputs=pred)
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    print('LSTM-CRF model created and compiled')
    return model

def create_lstm_crf_char_model():
    print('creating new LSTM-CRF-CHAR model')

    forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    forward_lstm_layer = LSTM(300, activation='relu')(forward_input)

    backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    backward_lstm_layer = LSTM(300, activation='relu')(backward_input)

    char_input = Input(shape=(CHAR_EMBEDDING_SIZE, ))
    char_lstm = Dense(200, activation='relu')(char_input)

    merge_one = concatenate([forward_lstm_layer, backward_lstm_layer])
    dropout = Dropout(0.6)(merge_one)
    merge_two = concatenate([dropout, char_lstm])
    dense = Dense(300, activation='relu')(merge_two)
    dropout2 = Dropout(0.6)(dense)
    dense2 = Dense(100, activation='relu')(dropout2)
    dense3 = Dense(NO_OF_CATEGORIES, activation='relu')(dense2)
    reshape = Reshape((1,9))(dense3)

    crf = CRF(NO_OF_CATEGORIES, sparse_target=False)
    pred = crf(reshape)
    model = Model(inputs=[forward_input, backward_input, char_input], outputs=pred)
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    print('LSTM-CRF-CHAR model created and compiled')
    return model

def create_lstm_char_model():
    print('creating new LSTM-CHAR model')

    forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    forward_lstm_layer = LSTM(300, activation='relu')(forward_input)

    backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    backward_lstm_layer = LSTM(300, activation='relu')(backward_input)

    char_input = Input(shape=(CHAR_EMBEDDING_SIZE,))
    char_dense = Dense(100, activation='relu')(char_input)
    char_drop = Dropout(0.3)(char_dense)

    merge_one = concatenate([forward_lstm_layer, backward_lstm_layer])
    dropout = Dropout(0.6)(merge_one)
    merge_two = concatenate([dropout, char_drop])
    dense = Dense(300, activation='relu')(merge_two)
    dropout2 = Dropout(0.6)(dense)
    dense2 = Dense(100, activation='relu')(dropout2)
    dense3 = Dense(NO_OF_CATEGORIES, activation='softmax')(dense2)

    model = Model(inputs=[forward_input, backward_input, char_input], outputs=dense3)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('LSTM-CHAR model created and compiled')
    return model



#GAN models, does not work


def create_generative_model():
    print('creating generative new model')

    forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    forward_lstm_layer = LSTM(128, activation='relu')(forward_input)

    backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    backward_lstm_layer = LSTM(128, activation='relu')(backward_input)

    merge_one = concatenate([forward_lstm_layer, backward_lstm_layer])
    dropout = Dropout(0.5)(merge_one)
    dense = Dense(64, activation='relu')(dropout)
    predictions = Dense(NO_OF_CATEGORIES, activation='softmax')(dense)

    model = Model(inputs=[forward_input, backward_input], outputs=predictions)
    optimizer = RMSprop(lr=0.004, clipvalue=1.0, decay=6e-8)
    model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print('generative model created and compiled')
    model.summary()
    return model


def create_discriminative_model():
    print('creating discriminative new model')
    forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    forward_lstm_layer = LSTM(128, activation='relu')(forward_input)

    backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    backward_lstm_layer = LSTM(128, activation='relu')(backward_input)

    label_input = Input(shape=(1,))

    merge_back_forw = concatenate([forward_lstm_layer, backward_lstm_layer])
    dropout = Dropout(0.5)(merge_back_forw)

    dense1 = Dense(32, activation='relu')(dropout)


    merge_data_label = concatenate([dense1,label_input])

    dense = Dense(16, activation='relu')(merge_data_label)

    fake_or_real_layer =Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[forward_input, backward_input, label_input], outputs=fake_or_real_layer)

    optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print('discriminative model created and compiled')
    model.summary()
    return model


def creat_gan_model(discriminator, generator):
    forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
    generator_layer = generator([forward_input, backward_input])
    discriminator_lable = discriminator([forward_input, backward_input,generator_layer])
    discriminator_lable.trainable = False
    GAN = Model(inputs=[forward_input, backward_input], outputs=discriminator_lable)

    optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    GAN.summary()
    return GAN


