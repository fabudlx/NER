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
        save_data_path = r'C:\Users\fkarl\PycharmProjects\NER\Model\Data\9cat\\' + data_set

        model_name = 'Char_Model'
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

        model_name = 'CRF_Char_Model'
        load_model_path = r'C:/Users/fkarl/PycharmProjects/NER/Model/NN_Models/9cat'

        w2v_file_name = 'de.wiki.bpe.op200000.d300.w2v.bin'


    LSTM = ModelsNN.create_lstm_char_model()
    save_load_utils.load_all_weights(LSTM, r'C:\Users\fkarl\PycharmProjects\NER\Model\NN_Models\9cat\\' + model_name)

    char_embedding_train = sl.load_padded_character_embedding_list(path=save_data_path,variable_name='train')
    chat_embedding_dev = sl.load_padded_character_embedding_list(path=save_data_path,variable_name='dev')

    # raw_data_dev = open(create_data_path + '.dev', 'r').read()
    # raw_data_dev = [word.split() for word in raw_data_dev.split('\n')]
    # chat_embedding_dev, max_len = ReadData.make_character_embeddings([line[0] for line in raw_data_dev if line])
    # sl.save_padded_character_embedding_list(chat_embedding_dev, 'dev', save_data_path)
    #
    #
    # raw_data_train = open(create_data_path + '.train', 'r').read()
    # raw_data_train = [word.split() for word in raw_data_train.split('\n')]
    # char_embedding_train, max_len = ReadData.make_character_embeddings([line[0] for line in raw_data_train if line])
    # sl.save_padded_character_embedding_list(char_embedding_train, 'train', save_data_path)

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


    char_embedding_train = np.reshape(char_embedding_train, (len(char_embedding_train), ModelsNN.CHAR_EMBEDDING_SIZE))
    chat_embedding_dev = np.reshape(chat_embedding_dev, (len(chat_embedding_dev), ModelsNN.CHAR_EMBEDDING_SIZE))
    Y_train = np.reshape(Y_train, (len(Y_train), ModelsNN.NO_OF_CATEGORIES))
    Y_dev = np.reshape(Y_dev, (len(Y_dev), ModelsNN.NO_OF_CATEGORIES))


    # char_embedding_train = np.reshape(char_embedding_train, (len(char_embedding_train), ModelsNN.CHAR_EMBEDDING_SIZE, 1))
    # chat_embedding_dev = np.reshape(chat_embedding_dev, (len(chat_embedding_dev), ModelsNN.CHAR_EMBEDDING_SIZE, 1))
    # Y_train = np.reshape(Y_train,(len(Y_train), 1, ModelsNN.NO_OF_CATEGORIES))
    # Y_dev = np.reshape(Y_dev,(len(Y_dev), 1, ModelsNN.NO_OF_CATEGORIES))

with tf.device('/gpu:0'):

    LSTM.fit([X_forward_train, X_backward_train, char_embedding_train], Y_train, batch_size=512, epochs=5, verbose=2)
    save_load_utils.save_all_weights(LSTM, r'C:\Users\fkarl\PycharmProjects\NER\Model\NN_Models\9cat\\'+model_name)

    prediction_softmax = LSTM.predict([X_forward_dev, X_backward_dev, chat_embedding_dev])

    prediction_list = [np.argmax(value) for value in prediction_softmax]

    target_list = [np.argmax(value) for value in Y_dev]

    precision, recall, fscore, support = metrics.precision_recall_fscore_support(target_list, prediction_list)

    print('Precision:',precision,'Mean:', np.mean(precision))
    print('Recall',recall,'Mean:', np.mean(recall))
    print('FScore:',fscore,'Mean:', np.mean(fscore))
    print('Detected:', Counter(prediction_list))
    print('Truth:', support)

    sl.write_results(target_list, prediction_list, result_file_name='CHAR_'+test_set+'_result_'+model_name, target_file_path =create_data_path +'.'+test_set, pos_of_tag=pos_of_tag)
