from Lib import queue
import numpy as np
import pickle
from Model import SaveAndLoad as sl, ModelsNN, ReadData

def create_crf(X_forward_data, X_backward_data, Y_data, model, mode, target_file_path, file_name ='crf_file' ):
    prediction_softmax_list = model.predict([X_forward_data, X_backward_data])
    Y_data_list = list(Y_data)

    return_prediction_softmax = []
    return_Y = []

    with open(target_file_path + '.' + mode, 'r') as target_file:

        target_file_lines = target_file.read().split('\n')

        spaces = 0
        for i in range(len(target_file_lines)):

            one_prediction_sample = []
            one_target_sample = []

            if not target_file_lines[i]:
                continue

            if i-1 != -1 and target_file_lines[i-1]:
                one_prediction_sample.append(np.array(prediction_softmax_list[i-1-spaces]))
                one_target_sample.append(Y_data_list[i-1-spaces])
            else:
                one_prediction_sample.append(np.zeros(ModelsNN.NO_OF_CATEGORIES)-1)
                one_target_sample.append(np.zeros(ModelsNN.NO_OF_CATEGORIES)-1)

            one_prediction_sample.append(np.array(prediction_softmax_list[i-spaces]))
            one_target_sample.append(Y_data_list[i-spaces])

            if i+1 != len(target_file_lines) and target_file_lines[i+1]:
                one_prediction_sample.append(np.array(prediction_softmax_list[i + 1 - spaces]))
                one_target_sample.append(Y_data_list[i + 1 - spaces])
            else:
                one_prediction_sample.append(np.zeros(ModelsNN.NO_OF_CATEGORIES) - 1)
                one_target_sample.append(np.zeros(ModelsNN.NO_OF_CATEGORIES) - 1)
                spaces += 1

            return_prediction_softmax.append(np.array(one_prediction_sample))
            return_Y.append(np.array(one_target_sample))


            # print('prediction',one_prediction_sample)
            # print('Y',one_target_sample)
        # print(return_Y[:10])
        # print(return_prediction_softmax[0],return_prediction_softmax[5],return_prediction_softmax[66])

        with open(r'C:\Users\fkarl\PycharmProjects\NER\Model\CRF_Input\\' + file_name + '_input.' + mode, 'wb') as input_crf:
            pickle.dump(np.array(return_prediction_softmax), input_crf, protocol=4)
        with open(r'C:\Users\fkarl\PycharmProjects\NER\Model\CRF_Input\\' + file_name + '_Y.' + mode, 'wb') as crf_Y:
            pickle.dump(np.array(return_Y), crf_Y, protocol=4)

        return return_prediction_softmax, return_Y



# def create_crf(X_forward_data, X_backward_data, Y_data, model, mode, target_file_path, file_name ='crf_file', ):
#
#     prediction_softmax = model.predict([X_forward_data, X_backward_data])
#     # prediction_softmax_list = list(prediction_softmax)
#     # Y_data_list = list(Y_data)
#
#     return_prediction_softmax = []
#     return_Y = []
#
#     input_sent = []
#     Y_sent = []
#
#     dt = np.dtype(float)
#     with open(target_file_path+'.'+mode, 'r') as target_file:
#
#         target_file_lines = target_file.read().split('\n')
#
#         spaces = 0
#         for i, target_line in enumerate(target_file_lines):
#
#
#             if target_line:
#                 # input_file.write(str(prediction_softmax_list[i - spaces]) + '\n')
#                 # Y_file.write(str(Y_data_list[i - spaces]) + '\n')
#
#                 input_sent.append(np.array(prediction_softmax[i - spaces],dtype=dt))
#                 Y_sent.append(np.array(Y_data[i - spaces],dtype=dt))
#             else:
#                 # input_file.write('\n')
#                 # Y_file.write('\n')
#                 spaces += 1
#
#                 return_prediction_softmax.append(np.array(input_sent))
#                 return_Y.append(np.array(Y_sent))
#
#                 input_sent.clear()
#                 Y_sent.clear()
#
#
#     # return_prediction_softmax = sequence.pad_sequences(return_prediction_softmax, maxlen=maxlen, value=-1,dtype='float64')
#     # return_Y = sequence.pad_sequences(return_Y, maxlen=maxlen, value=-1)
#     with open(r'../Model/CRF_Input/' + file_name + '_input.' + mode, 'wb') as input_crf:
#         pickle.dump(np.array(return_prediction_softmax), input_crf, protocol=4)
#     with open(r'../Model/CRF_Input/' + file_name + '_Y.' + mode, 'wb') as crf_Y:
#         pickle.dump(np.array(return_Y), crf_Y, protocol=4)
#
#
#     return np.array(return_prediction_softmax), np.array(return_Y)

def load_crf(file_name, mode):
    with open(r'C:\Users\fkarl\PycharmProjects\NER\Model\CRF_Input\\' + file_name + '_input.' + mode, 'rb') as input_crf_file:
        crf_input_tmp = pickle.load(input_crf_file)

    with open(r'C:\Users\fkarl\PycharmProjects\NER\Model\CRF_Input\\' + file_name + '_Y.' + mode, 'rb') as crf_Y_file:
        crf_Y_tmp = pickle.load(crf_Y_file)

    return crf_input_tmp, crf_Y_tmp



language = 'en'
test_set = 'dev'


if language == 'en':
    pos_of_tag = 3
    create_data_path = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_eng_IBO2'
    data_set = 'connl03_fasttext_IBO2_en'
    save_data_path = r'..\Model\Data\9cat\\' + data_set

    model_name = 'bi_lstm_fasttext_IBO2_en'
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

    model_name = 'bi_lstm_BPE_de'
    load_model_path = r'C:/Users/fkarl/PycharmProjects/NER/Model/NN_Models/9cat'

    w2v_file_name = 'de.wiki.bpe.op200000.d300.w2v.bin'


# LSTM = sl.loadModel(model_name, load_model_path)
# LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

##### DEV ###########
# ***** creating DEV data and saving it on disk *******
# X_forward_dev, X_backward_dev, Y_dev, words_unkown = ReadData.make_data_connl03(create_data_path + '.dev', w2v_class=w2v_class, pos_of_tag=pos_of_tag)
# sl.save_data(X_forward_dev,X_backward_dev, Y_dev, 'dev', save_data_path)

# ***** loading DEV data from disk *******
X_forward_dev, X_backward_dev, Y_dev = sl.load_data(data_set=data_set,variable_name=test_set)

# print(Y_dev)

# crf_input, crf_Y = create_crf(X_forward_dev, X_backward_dev, Y_dev, model = LSTM, mode='dev', target_file_path= create_data_path, file_name =data_set + '_crf')
crf_input, crf_Y = load_crf(file_name =data_set + '_crf', mode='dev')

crf_input = np.array(crf_input)
int_encoing = []
for triples in crf_Y:
    triple_temp =[]
    for hot_one in triples:
        if -1 in hot_one:
            triple_temp.append(np.array([-1,]))
        else:
            triple_temp.append(np.array([np.argmax(hot_one),]))
    int_encoing.append(np.array(triple_temp))

# crf_Y = [np.argmax(categories) in triple for triple in crf_Y for categories in triple ]
# crf_Y = np.array(crf_input)
print(int_encoing[:5])
int_encoing = np.reshape(np.array(int_encoing),(len(int_encoing),3,1))
# print(crf_input.shape)
# print(crf_Y[:20])

CRF = ModelsNN.create_crf_model()

CRF.fit([crf_input], int_encoing, batch_size=256, epochs=10)

prediction = CRF.predict([crf_input])
print(prediction[:10])

crf_prediction = [sample[1] for sample in prediction]
prediction_list = [np.argmax(value) for value in crf_prediction]


target_list = [np.argmax(value) for value in Y_dev]


# print(crf_input.shape)
# print(seq_length.shape)
# print(crf_Y.shape)
# from collections import defaultdict
# samples = defaultdict(list)
# for x,l,y in zip(crf_input,seq_length,crf_Y):
#     samples[l].append([x,y])
#
# print(samples[2])
#
# for l, sample in samples.items():
#     x = [x[0] for x in sample]
#     y = [y[1] for y in sample]
#     length = np.array([l]*len(x))
#     print(length)
#     length = np.reshape(length,(len(length),1))
#     print(length.shape,np.array(x).shape,np.array(y).shape)
#     CRF.fit([np.array(x),length],np.array(y))
#
# prediction_softmax = CRF.predict([crf_input, seq_length])
# prediction_list = [np.argmax(value) for value in prediction_softmax]
# print(prediction_list)
#
# target_list = [np.argmax(value) for value in crf_Y]
# print(target_list)
#
sl.write_results(target_list, prediction_list, result_file_name='CRF_'+test_set+'_result_'+model_name, target_file_path =create_data_path +'.'+test_set, pos_of_tag=pos_of_tag)