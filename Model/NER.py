from collections import Counter
import numpy as np
from sklearn import metrics
from keras_contrib.utils import save_load_utils
from Model import SaveAndLoad as sl, ModelsNN
import os.path
import os
from Model import ReadData
import zstd
import re



#TODO: replace with your path
raw_data_path = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\\'
model_folder_path = r'..\Resources\NN_Models\\'
data_folder_path = r'..\Resources\Data\\'


def create_training_data(w2v_class, embedding_model, data_set ='connl03', data_format ='IBO2', language ='eng', pos_of_tag = 3):

    path_to_save = data_folder_path+data_set+'\\' + embedding_model

    if data_set == 'connl03':
        path_to_raw_data = raw_data_path + data_set + '\\' + data_format + '_' + language
        data_sets = ['dev','train', 'test']
        for ending in data_sets:
            for file in os.listdir(path_to_raw_data):
                if file.endswith(ending):
                    X_forward, X_backward, Y, char_embedding, words_unkown = ReadData.make_data_connl03(path_to_raw_data+'\\'+file, w2v_class=w2v_class, pos_of_tag=pos_of_tag)
                    compress_and_save(X_forward, X_backward, Y, char_embedding, path_to_save, ending)

    elif data_set == 'germeval':
        path_to_raw_data = raw_data_path + data_set
        data_sets = ['dev','train', 'test']
        for ending in data_sets:
            for file in os.listdir(path_to_raw_data):
                if file.endswith(ending):
                    X_forward, X_backward, Y, char_embedding, words_unkown = ReadData.make_data_germEval(path_to_raw_data + '\\' + file, w2v_class=w2v_class, pos_of_tag=pos_of_tag)
                    compress_and_save(X_forward,X_backward,Y,char_embedding,path_to_save,ending)
    else:
        print('data set not found!')

def compress_and_save(X_forward,X_backward,Y,char_embedding,path_to_save,ending):
    print('compressing data')
    cctx = zstd.ZstdCompressor(write_content_size=True)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    X_forward_compressed = cctx.compress(X_forward)
    X_backward_compressed = cctx.compress(X_backward)
    print('saving data')
    sl.save_data(X_forward_compressed, X_backward_compressed, Y, ending, path_to_save)
    sl.save_padded_character_embedding_list(char_embedding, ending, path_to_save)



def train_model(model_type, model_name, batch_size=512, epochs = 1, data_set = 'connl03', embedding_model='fasttext_IBO2_en'):

    dctx = zstd.ZstdDecompressor()
    model_types = {'bi-lstm': ModelsNN.create_lstm_model,
                   'bi-lstm_crf':ModelsNN.create_lstm_crf_model,
                   'bi-lstm_char':ModelsNN.create_lstm_char_model,
                   'bi-lstm_crf_char':ModelsNN.create_lstm_crf_char_model,
                   'cnn_char':ModelsNN.create_cnn_char_model}



    if model_type in model_types:
        print('Model type', model_type, 'detected')

        LSTM = model_types[model_type]()
        print('Model was initialized')

        model_path = model_folder_path + data_set + '\\'+model_name

        if (os.path.isfile(model_path)):
            save_load_utils.load_all_weights(LSTM, model_path)
            print('Weights from old model found and loaded')
        else:
            print('Model is traind from scrach')


        path_to_data = data_folder_path+data_set+'\\' + embedding_model



        X_forward_train, X_backward_train, Y_train = sl.load_data(path=path_to_data, variable_name='train')
        if type(X_forward_train) == bytes:
            print('decompressing data')
            X_forward_train = dctx.decompress(X_forward_train)
            X_forward_train = np.fromstring(X_forward_train)
            X_forward_train = np.reshape(X_forward_train, (-1, ModelsNN.SENTENCE_LENGTH, ModelsNN.EMBEDDING_SIZE))

            X_backward_train = dctx.decompress(X_backward_train)
            X_backward_train = np.fromstring(X_backward_train)
            X_backward_train = np.reshape(X_backward_train, (-1, ModelsNN.SENTENCE_LENGTH, ModelsNN.EMBEDDING_SIZE))

        char_embedding_train = sl.load_padded_character_embedding_list(path=path_to_data, variable_name='train')
        print('data was decompressed and is ready')
        print('forward',X_forward_train.shape)
        # print('backward', X_backward_train.shape)
        # print('Y', Y_train.shape)
        # print('embeddings', char_embedding_train.shape)

        name_to_save = model_type+'_'+data_set+'_' + embedding_model
        path_to_save = model_folder_path + data_set + '\\'+name_to_save

        if not os.path.exists(model_folder_path + data_set):
            os.makedirs(model_folder_path + data_set)

        if model_type == 'bi-lstm' :

            LSTM.fit([X_forward_train, X_backward_train], Y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True)
            save_load_utils.save_all_weights(LSTM, path_to_save)

        elif model_type == 'bi-lstm_crf':

            Y_train = np.reshape(Y_train, (len(Y_train), 1, ModelsNN.NO_OF_CATEGORIES))

            LSTM.fit([X_forward_train, X_backward_train], Y_train, batch_size=batch_size, epochs=epochs, verbose=2)
            save_load_utils.save_all_weights(LSTM, path_to_save)


        elif model_type == 'bi-lstm_char' or model_type == 'cnn_char':

            LSTM.fit([X_forward_train, X_backward_train, char_embedding_train], Y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True)
            save_load_utils.save_all_weights(LSTM, path_to_save)

        elif model_type == 'bi-lstm_crf_char':

            Y_train = np.reshape(Y_train, (len(Y_train), 1, ModelsNN.NO_OF_CATEGORIES))

            LSTM.fit([X_forward_train, X_backward_train, char_embedding_train], Y_train, batch_size=batch_size, epochs=epochs, verbose=2)
            save_load_utils.save_all_weights(LSTM, path_to_save)




def test_model(model_type, model_name, test_set_ending ='dev', data_set ='connl03', embedding_model='fasttext_IBO2_en', data_format ='IBO2', language ='eng', pos_of_tag = 3, pos_of_word=0, save_name = None):
    dctx = zstd.ZstdDecompressor()
    model_types = {'bi-lstm': ModelsNN.create_lstm_model,
                   'bi-lstm_crf': ModelsNN.create_lstm_crf_model,
                   'bi-lstm_char': ModelsNN.create_lstm_char_model,
                   'bi-lstm_crf_char': ModelsNN.create_lstm_crf_char_model,
                   'cnn_char':ModelsNN.create_cnn_char_model}

    if model_type in model_types:
        print('Model type', model_type, 'detected')

        LSTM = model_types[model_type]()
        print('Model was initialized')

        model_path = model_folder_path + data_set + '\\' + model_name

        if (os.path.isfile(model_path)):
            save_load_utils.load_all_weights(LSTM, model_path)
            print('Weights from old model found and loaded')
        else:
            print('No weights to load found!')

        path_to_data = data_folder_path + data_set + '\\' + embedding_model

        X_forward_test, X_backward_test, Y_test = sl.load_data(path=path_to_data, variable_name=test_set_ending)
        if type(X_forward_test) == bytes:
            print('decompressing data')
            X_forward_test = dctx.decompress(X_forward_test)
            X_forward_test = np.fromstring(X_forward_test)
            X_forward_test = np.reshape(X_forward_test, (-1, ModelsNN.SENTENCE_LENGTH, ModelsNN.EMBEDDING_SIZE))

            X_backward_test = dctx.decompress(X_backward_test)
            X_backward_test = np.fromstring(X_backward_test)
            X_backward_test = np.reshape(X_backward_test, (-1, ModelsNN.SENTENCE_LENGTH, ModelsNN.EMBEDDING_SIZE))

        char_embedding_test = sl.load_padded_character_embedding_list(path=path_to_data, variable_name=test_set_ending)
        print('test data loaded')
        print('forward',X_forward_test.shape)
        if save_name is None:
            name_to_save = model_type+'_'+data_set+'_' + embedding_model
        else:
            name_to_save = save_name
        path_to_save = model_folder_path + data_set + '\\'+name_to_save

        if model_type == 'bi-lstm' or model_type == 'bi-lstm_crf':

            prediction_softmax = LSTM.predict([X_forward_test, X_backward_test])

        else:

            prediction_softmax = LSTM.predict([X_forward_test, X_backward_test, char_embedding_test])


        prediction_list = [np.argmax(value) for value in prediction_softmax]

        target_list = [np.argmax(value) for value in Y_test]

        precision, recall, fscore, support = metrics.precision_recall_fscore_support(target_list, prediction_list)

        print('Precision:', precision, 'Mean:', np.mean(precision))
        print('Recall', recall, 'Mean:', np.mean(recall))
        print('FScore:', fscore, 'Mean:', np.mean(fscore))
        print('Detected:', Counter(prediction_list))
        print('Truth:', support)

        if data_set =='connl03':
            path_to_raw_data = raw_data_path + data_set + '\\' + data_format + '_' + language
        elif data_set =='germeval':
            path_to_raw_data = raw_data_path + data_set
        else:
            print('wrong data_set name!!!')

        for file in os.listdir(path_to_raw_data):
            if file.endswith(test_set_ending):
                sl.write_results(target_list, prediction_list,data_set= data_set, result_file_name= test_set_ending.capitalize() + '_result_' + name_to_save, target_file_path=path_to_raw_data+'\\'+file, pos_of_tag=pos_of_tag, pos_of_word=pos_of_word)

def tag_sentence(w2v_class, sentence, model_type='bi-lstm', model_name='bi-lstm_connl03_fasttext_deu',data_set='connl03'):

    model_types = {'bi-lstm': ModelsNN.create_lstm_model,
                   'bi-lstm_crf': ModelsNN.create_lstm_crf_model,
                   'bi-lstm_char': ModelsNN.create_lstm_char_model,
                   'bi-lstm_crf_char': ModelsNN.create_lstm_crf_char_model}

    if model_type in model_types:
        print('Model type', model_type, 'detected')

        LSTM = model_types[model_type]()
        print('Model was initialized')

        model_path = model_folder_path + data_set + '\\' + model_name

        if (os.path.isfile(model_path)):
            save_load_utils.load_all_weights(LSTM, model_path)
            print('Weights from old model found and loaded')
        else:
            print('No model found in', model_path)
            return None

        sentence = ReadData.pad_punctuation(sentence).strip().split(' ')

        print('reading sentence')
        X_forward = []
        X_backward = []

        previous_words_from_sentence = []
        for word in sentence:
            previous_words_from_sentence.append(word)
            # print(previous_words_from_sentence)
            embedded_forward_sentence, unknown = w2v_class.get_embedding_improved(previous_words_from_sentence)
            padded_forward_sentence = ReadData.pad_in_front(embedded_forward_sentence)
            X_forward.append(padded_forward_sentence)

        previous_words_from_sentence = list(reversed(previous_words_from_sentence))
        while previous_words_from_sentence:
            # print(previous_words_from_sentence)
            embedded_backwards_sentence, unknown = w2v_class.get_embedding_improved(previous_words_from_sentence)
            padded_backwards_sentence = ReadData.pad_in_front(embedded_backwards_sentence)
            X_backward.append(padded_backwards_sentence)
            previous_words_from_sentence.pop()

        X_forward = np.array(X_forward)
        X_backward = np.array(X_backward)

        char_embedding = ReadData.make_character_embeddings(sentence, ModelsNN.CHAR_EMBEDDING_SIZE)
        if model_type == 'bi-lstm' or model_type == 'bi-lstm_crf':
            prediction_softmax = LSTM.predict([X_forward, X_backward])
        else:  # model_type == 'bi-lstm_char' or model_type == 'bi-lstm_crf_char':
            prediction_softmax = LSTM.predict([X_forward, X_backward, char_embedding])

        prediction_list = [sl.get_tag(np.argmax(value)) for value in prediction_softmax]
        tagged_sentence = zip(sentence, prediction_list)
        # print(tagged_sentence)
        return(tagged_sentence)

    else:
        print('model unknown!!!')
        return None