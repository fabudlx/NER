from Model import ModelsNN, SaveAndLoad as sl
from Model.ModelsNN import NO_OF_CATEGORIES
from random import randint
import numpy as np
from sklearn import metrics
from collections import Counter
from Model import SaveAndLoad as sl
from keras.optimizers import RMSprop

BATCH_SIZE = 1000
COUNTER = 0

def get_tag(tag_no):
    if tag_no == 0:
        return 'O'
    elif tag_no == 1:
        return 'I-ORG'
    elif tag_no == 2:
        return 'I-LOC'
    elif tag_no == 3:
        return 'I-PER'
    elif tag_no == 4:
        return 'I-MISC'
    else:
        raise EnvironmentError

def make_trainable(network, is_trainable):
    network.trainable = is_trainable
    for l in network.layers:
        l.trainable = is_trainable

def get_data():
    global COUNTER
    COUNTER += randint(0, 5000)

    max_elements_per_category = int(BATCH_SIZE / NO_OF_CATEGORIES)

    X_forward_batch = []
    X_backward_batch = []
    Y_batch_true = []
    category_counter = {i: 0 for i in range(NO_OF_CATEGORIES)}
    Y_number = [np.argmax(value) for value in Y_train]
    print(Counter(Y_number))
    while len(Y_batch_true) < BATCH_SIZE :

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
    return X_forward_batch, X_backward_batch, Y_batch_true




# generator = ModelsNN.create_generative_model()

generator = sl.loadModel('bi_model3')
generator.name = 'bi_model'
# generator.summery()
optimizer = RMSprop(lr=0.004, clipvalue=1.0, decay=6e-8)
# generator.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


discriminator = ModelsNN.create_discriminative_model()

gan_model = ModelsNN.creat_gan_model(discriminator, generator)


X_forward_train, X_backward_train, Y_train = sl.load_data('train')


epochs = 1000

for i in range(epochs):

    X_forward_batch, X_backward_batch, Y_batch_true = get_data()

    Y_batch_fake = generator.predict([np.array(X_forward_batch), np.array(X_backward_batch)])
    Y_bach_fake_temp = np.array([np.argmax(values) for values in Y_batch_fake]).reshape(-1)
    Y_bach_fake_one_hot = np.eye(NO_OF_CATEGORIES)[Y_bach_fake_temp]

    X_forward_batch = np.array(X_forward_batch+X_forward_batch)
    X_backward_batch = np.array(X_backward_batch+X_backward_batch)
    Y_batch = np.concatenate( (np.array(Y_batch_true), Y_bach_fake_one_hot), axis=0)
    print(Y_batch.shape)

    Y = np.ones(BATCH_SIZE * 2)
    Y[BATCH_SIZE:] = 0
    make_trainable(discriminator, True)
    print('---------------- train Discriminator -------------------')

    #TODO shuffel data frist
    discriminator.fit([X_forward_batch, X_backward_batch, Y_batch], Y, batch_size=100, epochs=7, shuffle=True)

    #jetzt 0,0,1,0,0 -> könnte auch 2
    target_list = [np.argmax(values) for values in Y_batch_true]
    prediction_list = [np.argmax(values) for values in Y_bach_fake_one_hot]



    # discriminator.fit([X_forward_batch, X_backward_batch, Y_bach_fake_one_hot], np.zeros(BATCH_SIZE), batch_size=128, epochs=2)

    if i % 1 == 0:
        #do statistics
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(target_list, prediction_list)
        print('Precision:', precision, 'Mean:', np.mean(precision))
        print('Recall', recall, 'Mean:', np.mean(recall))
        print('FScore:', fscore, 'Mean:', np.mean(fscore))
        print('Detected:', Counter(prediction_list))
        print('Truth:', support)




    #train generator

    X_forward_batch, X_backward_batch, _ = get_data()

    X_forward_batch = np.array(X_forward_batch)
    X_backward_batch = np.array(X_backward_batch)

    make_trainable(discriminator,False)
    # disc_prediction = discriminator.predict([X_forward_batch, X_backward_batch, one_hot_predictions])
    print('---------------- train Generator -------------------')
    gan_model.fit([X_forward_batch, X_backward_batch], np.ones(BATCH_SIZE), epochs=3, batch_size=100)