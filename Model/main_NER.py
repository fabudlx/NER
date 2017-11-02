from collections import Counter
import numpy as np
from sklearn import metrics
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils
from Model import SaveAndLoad as sl, ModelsNN
from Model.WordEmbeddings import WordEmbedding

from Model import ReadData
from keras.preprocessing import sequence
import tensorflow as tf
from Lib import queue
from keras.models import load_model






with tf.device('/cpu:0'):
    language = 'en'
    test_set = 'dev'


    if language == 'en':
        pos_of_tag = 3
        create_data_path = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_eng_IBO2'
        data_set = 'connl03_fasttext_IBO2_en'
        save_data_path = r'..\Model\Data\9cat\\' + data_set

        model_name = 'CRF_Model_300'
        load_model_path = r'C:/Users/fkarl/PycharmProjects/NER/Model/NN_Models/9cat'

        w2v_file_name = 'wiki.en.vec'
        binary=False
        # w2v_file_name = 'en.wiki.bpe.op200000.d300.w2v.bin'
        # binary = True

    elif language == 'de':
        pos_of_tag = 4
        create_data_path = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_deu_IBO2'
        save_data_path = r'..\Model\Data\9cat\connl03_BPEmb_de'
        data_set = 'connl03_BPEmb_de'

        model_name = 'CRF_Model'
        load_model_path = r'C:/Users/fkarl/PycharmProjects/NER/Model/NN_Models/9cat'

        w2v_file_name = 'de.wiki.bpe.op200000.d300.w2v.bin'



    # LSTM = load_model(r'C:\Users\fkarl\PycharmProjects\NER\Model\NN_Models\9cat\\' + model_name)
    # LSTM.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

    LSTM = ModelsNN.create_lstm_crf_model()
    # save_load_utils.load_all_weights(LSTM, r'C:\Users\fkarl\PycharmProjects\NER\Model\NN_Models\9cat\\' + model_name)



    # w2v_class = WordEmbedding.EmbeddingModel(path =r'C:\Users\fkarl\Desktop\Science Stuff\pretrained Model\\' + w2v_file_name, binary = binary) #path=r'C:\Users\fkarl\Desktop\Science Stuff\pretrained Model\german.model', binary=binary

    ###### TRAIN #########
    # ***** creating TRAIN data and saving it on disk *******
    # X_forward_train, X_backward_train, Y_train, words_unkown = ReadData.make_data_connl03(create_data_path + '.train', w2v_class=w2v_class, pos_of_tag=pos_of_tag)
    # sl.save_data(X_forward_train,X_backward_train, Y_train, 'train', save_data_path)

    # ***** loading TRAIN data from disk *******
    X_forward_train, X_backward_train, Y_train = sl.load_data(data_set=data_set,variable_name='train')

    ###### DEV ###########
    # ***** creating DEV data and saving it on disk *******
    # X_forward_dev, X_backward_dev, Y_dev, words_unkown = ReadData.make_data_connl03(create_data_path + '.dev', w2v_class=w2v_class, pos_of_tag=pos_of_tag)
    # sl.save_data(X_forward_dev,X_backward_dev, Y_dev, 'dev', save_data_path)

    # ***** loading DEV data from disk *******
    X_forward_dev, X_backward_dev, Y_dev = sl.load_data(data_set=data_set,variable_name=test_set)

    ###### TEST ###########
    # ***** creating TEST data and saving it on disk *******
    # X_forward_test, X_backward_test, Y_test, words_unkown = ReadData.make_data_connl03(create_data_path + '.test', w2v_class=w2v_class, pos_of_tag=pos_of_tag)
    # sl.save_data(X_forward_test,X_backward_test, Y_test,'test',save_data_path)

    # ***** loading TEST data from disk *******
    # X_forward_test, X_backward_test, Y_test = sl.load_data(data_set=data_set, variable_name='test')

    # print(words_unkown)
    # print(len(words_unkown))


    Y_train = np.reshape(Y_train,(len(Y_train), 1, ModelsNN.NO_OF_CATEGORIES))
    Y_dev = np.reshape(Y_dev,(len(Y_dev), 1, ModelsNN.NO_OF_CATEGORIES))
with tf.device('/gpu:0'):
    # LSTM.fit([X_forward_train, X_backward_train], Y_train, batch_size=10, epochs=2, verbose=0)
    # LSTM.fit([X_forward_train, X_backward_train],Y_train, batch_size=30, epochs=2,verbose=0)
    LSTM.fit([X_forward_train, X_backward_train], Y_train, batch_size=300, epochs=10, verbose=2)
    save_load_utils.save_all_weights(LSTM, r'C:\Users\fkarl\PycharmProjects\NER\Model\NN_Models\9cat\\'+model_name)

    # LSTM.save(r'C:\Users\fkarl\PycharmProjects\NER\Model\NN_Models\9cat\\'+model_name)



    prediction_softmax = LSTM.predict([X_forward_dev, X_backward_dev])

    prediction_list = [np.argmax(value) for value in prediction_softmax]

    target_list = [np.argmax(value) for value in Y_dev]

    precision, recall, fscore, support = metrics.precision_recall_fscore_support(target_list, prediction_list)

    print('Precision:',precision,'Mean:', np.mean(precision))
    print('Recall',recall,'Mean:', np.mean(recall))
    print('FScore:',fscore,'Mean:', np.mean(fscore))
    print('Detected:', Counter(prediction_list))
    print('Truth:', support)

    sl.write_results(target_list, prediction_list, result_file_name='CRF_'+test_set+'_result_'+model_name, target_file_path =create_data_path +'.'+test_set, pos_of_tag=pos_of_tag)

    #     # ***** fit data like it is *******
    #     # LSTM.fit([X_forward_train, X_backward_train], Y_train, batch_size=1024, epochs=15, shuffle=True)#, validation_data=([X_forward_dev,X_backward_dev],Y_dev)
    #     # sl.save_model(LSTM, model_name)
    #
    #

# ***** fit data with same number of datapoints per category *******
# for i in range(3):
#     print('>>> Epoch ',i)
#     X_forward_train_batch, X_backward_train_batch, Y_train_batch = ReadData.get_data_regulized(X_forward_train, X_backward_train, Y_train, bach_size=5000)
#     LSTM.fit([X_forward_train_batch, X_backward_train_batch], Y_train_batch, batch_size=256, epochs=1, shuffle=True)
#
# sl.save_model(LSTM, 'bi_model_same_cat_size')


# score = LSTM.evaluate(X_test, Y_test, batch_size=512)
# print(score)

# count = 0
# for target in Y_test:
#     if target[0] == 1:
#         count += 1
#
# print(count/len(Y_test))



# forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
# forward_lstm_layer = LSTM(128, activation='relu')(forward_input)
# forward_dropout = Dropout(0.5)(forward_lstm_layer)
# forward_dense = Dense(32, activation='relu')(forward_dropout)
# backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
# backward_lstm_layer = LSTM(128, activation='relu')(backward_input)
# backward_dropout = Dropout(0.5)(backward_lstm_layer)
# backward_dense = Dense(32,activation='relu')(backward_dropout)
# merge_one = concatenate([forward_lstm_layer, backward_lstm_layer])
# predictions = Dense(5, activation='softmax')(dense)
# model = Model(inputs=[forward_input, backward_input], outputs=predictions)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Precision: [ 0.96059463  0.82390511  0.83070636  0.85319315  0.76062323] Mean: 0.845804494911
# Recall [ 0.98384085  0.72355769  0.73922078  0.790119    0.58496732] Mean: 0.764341128617
# FScore: [ 0.97207878  0.77047782  0.78229797  0.82044561  0.66133005] Mean: 0.801326043883
# Detected: Counter({0: 39487, 3: 2568, 1: 2192, 2: 1713, 4: 706})
# Truth: [38554  2496  1925  2773   918]


# Precision: [ 0.96242811  0.82909415  0.82566723  0.88461538  0.73997234] Mean: 0.84835544456
# Recall [ 0.98531929  0.74439103  0.75532468  0.78795528  0.58278867] Mean: 0.7711557895
# FScore: [ 0.97373919  0.78446274  0.78893109  0.83349228  0.65204144] Mean: 0.806533346093
# Detected: Counter({0: 39471, 3: 2470, 1: 2241, 2: 1761, 4: 723})
# Truth: [38554  2496  1925  2773   918]



# forward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
# forward_lstm_layer = LSTM(128, activation='relu')(forward_input)
# backward_input = Input(shape=(SENTENCE_LENGTH, EMBEDDING_SIZE))
# backward_lstm_layer = LSTM(128, activation='relu')(backward_input)
# merge_one = concatenate([forward_lstm_layer, backward_lstm_layer])
# dropout = Dropout(0.5)(merge_one)
# dense = Dense(64, activation='relu')(dropout)
# predictions = Dense(5, activation='softmax')(dense)
# model = Model(inputs=[forward_input, backward_input], outputs=predictions)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# Precision: [ 0.96904112  0.84482759  0.83590016  0.85960314  0.77716644] Mean: 0.85730768894
# Recall [ 0.98480054  0.74599359  0.81766234  0.82798413  0.61546841] Mean: 0.798381801841
# FScore: [ 0.97685727  0.79234043  0.82668067  0.84349743  0.68693009] Mean: 0.825261177769
# Detected: Counter({0: 39181, 3: 2671, 1: 2204, 2: 1883, 4: 727})
# Truth: [38554  2496  1925  2773   918]


# same model - fasttext

# Precision: [ 0.97552699  0.77846028  0.86677282  0.91912039  0.79128137] Mean: 0.866232371199
# Recall [ 0.98428179  0.76161859  0.84831169  0.88928958  0.65250545] Mean: 0.827201417868
# FScore: [ 0.97988483  0.76994735  0.8574429   0.90395894  0.71522388] Mean: 0.84529158105
# Detected: Counter({0: 38900, 3: 2683, 1: 2442, 2: 1884, 4: 757})
# Truth: [38554  2496  1925  2773   918]

#test data

# Precision: [ 0.97086849  0.84758155  0.91431557  0.94825269  0.9135539 ] Mean: 0.918914438572
# Recall [ 0.99341478  0.72036329  0.83572111  0.89615751  0.67507886] Mean: 0.824147109471
# FScore: [ 0.98201224  0.77881137  0.87325349  0.92146939  0.77641723] Mean: 0.866392744207
# Detected: Counter({0: 43973, 1: 1778, 2: 1914, 3: 2976, 4: 937})
# Truth: [42975  2092  2094  3149  1268]


#same category size

# bi_model_same_cat_size saved model to disk
# test data was loaded
# Precision: [ 0.99214155  0.60574876  0.73712949  0.83674614  0.40650711] Mean: 0.71565460877
# Recall [ 0.92246655  0.81596558  0.9025788   0.94728485  0.85725552] Mean: 0.889110260579
# FScore: [ 0.95603627  0.69531568  0.81150708  0.888591    0.5514967 ] Mean: 0.78058934872
# Detected: Counter({0: 39957, 3: 3565, 1: 2818, 4: 2674, 2: 2564})
# Truth: [42975  2092  2094  3149  1268]

# bi_model_bigger saved model to disk
# test data was loaded
# Precision: [ 0.97587297  0.87160208  0.87033582  0.94593698  0.89961759] Mean: 0.912673089567
# Recall [ 0.99106457  0.72036329  0.89111748  0.90568434  0.74211356] Mean: 0.850068649712
# FScore: [ 0.98341011  0.78879874  0.88060406  0.92537313  0.81331029] Mean: 0.878299265636
# Detected: Counter({0: 43644, 3: 3015, 2: 2144, 1: 1729, 4: 1046})
# Truth: [42975  2092  2094  3149  1268]


# bi_model_bigger_high_drop saved model to disk
# test data was loaded
# Precision: [ 0.97753608  0.84516129  0.89414634  0.9632778   0.85602094] Mean: 0.90722849062
# Recall [ 0.99132054  0.75143403  0.87535817  0.89965068  0.77365931] Mean: 0.85828454491
# FScore: [ 0.98438005  0.79554656  0.88465251  0.93037767  0.81275891] Mean: 0.881543139515
# Detected: Counter({0: 43581, 3: 2941, 2: 2050, 1: 1860, 4: 1146})
# Truth: [42975  2092  2094  3149  1268]


# bi_model_cnn saved model to disk
# Precision: [ 0.98552735  0.84850025  0.92226991  0.97030342  0.86230637] Mean: 0.917781460711
# Recall [ 0.99192554  0.79780115  0.92359121  0.95458876  0.79022082] Mean: 0.891625495369
# FScore: [ 0.98871609  0.82237004  0.92293009  0.96238194  0.82469136] Mean: 0.904217904944
# Detected: Counter({0: 43254, 3: 3098, 2: 2097, 1: 1967, 4: 1162})
# Truth: [42975  2092  2094  3149  1268]


# bi_lstm_de saved model to disk
# Precision: [ 0.98344165  0.92062459  0.82873319  0.94287185  0.65891473] Mean: 0.866917202093
# Recall [ 0.99600304  0.72638604  0.87518685  0.91645823  0.26153846] Mean: 0.755114522963
# FScore: [ 0.98968249  0.81205165  0.85132679  0.92947742  0.37444934] Mean: 0.791397538062
# Detected: Counter({0: 46623, 3: 1943, 1: 1537, 2: 1413, 4: 129})
# Truth: [46035  1948  1338  1999   325]