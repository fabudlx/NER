from keras.models import model_from_json
import pickle


def save_model(model, name, path =r'C:/Users/fkarl/PycharmProjects/NER/Model/NN_Models/9cat/'):
    model_json = model.to_json()
    with open(path+name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+name+".h5")
    print(name+ " saved model to disk")


def loadModel(name,  path =r'C:/Users/fkarl/PycharmProjects/NER/Model/NN_Models/9cat'):
    # load json and create model
    json_file = open(path+'/'+name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+'/'+name + ".h5")
    print("Loaded model from disk")
    return loaded_model


def load_data(path = r'C:\Users\fkarl\PycharmProjects\NER\Model\Data\9cat', data_set = 'connl03' , variable_name = 'train'):

    with open(path+'/'+data_set+r'\X_forward.'+variable_name, 'rb') as f:
        X_forward = pickle.load(f)

    with open(path+'/'+data_set+r'\X_backward.'+variable_name, 'rb') as f:
        X_backward = pickle.load(f)

    with open(path+'/'+data_set+r'\Y.'+variable_name, 'rb') as f:
        Y = pickle.load(f)

    print(variable_name,'data was loaded')
    return X_forward, X_backward, Y


def save_data(X_forward, X_backward, Y, variable_name, path = '..\Model\Data\9cat\connl03_BPEmb_en'):
    print('saving data in',path,'under the name',variable_name)

    with open(path+ r'\X_forward.'+variable_name, 'wb') as f:
        pickle.dump(X_forward, f,protocol=4)
    with open(path+ r'\X_backward.'+variable_name, 'wb') as f:
        pickle.dump(X_backward, f,protocol=4)
    with open(path+ r'\Y.'+variable_name, 'wb') as f:
        pickle.dump(Y, f,protocol=4)

    print('data saved on disk')

def save_padded_character_embedding_list(padded_character_embedding_list, variable_name, path = '..\Model\Data\9cat\connl03_BPEmb_en'):
    with open(path+ r'\char_embedding.'+variable_name, 'wb') as f:
        pickle.dump(padded_character_embedding_list, f, protocol=4)

def load_padded_character_embedding_list(path = r'C:\Users\fkarl\PycharmProjects\NER\Model\Data\9cat' , variable_name = 'train'):
    with open(path+r'\char_embedding.'+variable_name, 'rb') as f:
        padded_character_embedding_list = pickle.load(f)
    return padded_character_embedding_list

def load_crf(file_name, mode):
    with open(r'C:\Users\fkarl\PycharmProjects\NER\Model\CRF_Input\\' + file_name + '_input.' + mode, 'rb') as input_crf_file:
        crf_input_tmp = pickle.load(input_crf_file)

    with open(r'C:\Users\fkarl\PycharmProjects\NER\Model\CRF_Input\\' + file_name + '_Y.' + mode, 'rb') as crf_Y_file:
        crf_Y_tmp = pickle.load(crf_Y_file)

    return crf_input_tmp, crf_Y_tmp




def get_tag(tag_no):
    if tag_no == 0:
        return 'O'
    elif tag_no == 1:
        return 'I-ORG'
    elif tag_no == 2:
        return 'B-ORG'
    elif tag_no == 3:
        return 'I-LOC'
    elif tag_no == 4:
        return 'B-LOC'
    elif tag_no == 5:
        return 'I-PER'
    elif tag_no == 6:
        return 'B-PER'
    elif tag_no == 7:
        return 'I-MISC'
    elif tag_no == 8:
        return 'B-MISC'
    else:
        raise EnvironmentError

# def get_tag(tag_no,tag_pos):
#     if tag_no == 0:
#         return 'O'
#     elif tag_no == 1:
#         if tag_pos == 1:
#             return 'B-ORG'
#         elif tag_pos == 0:
#             return 'I-ORG'
#     elif tag_no == 2:
#         if tag_pos == 1:
#             return 'B-LOC'
#         elif tag_pos == 0:
#             return 'I-LOC'
#     elif tag_no == 3:
#         if tag_pos == 1:
#             return 'B-PER'
#         elif tag_pos == 0:
#             return 'I-PER'
#     elif tag_no == 4:
#         if tag_pos == 1:
#             return 'B-MISC'
#         elif tag_pos == 0:
#             return 'I-MISC'
#     else:
#         raise EnvironmentError

def write_results(target_list, prediction_list, data_set = '9cat/connl03/', result_file_name='results_unnamed', target_file_path =r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_eng_IBO2.test', pos_of_tag = 3, pos_of_word = 0):
    with open(r'C:\Users\fkarl\PycharmProjects\NER\Model\Results\\'+data_set + result_file_name, 'w') as result_file:
        with open(target_file_path, 'r') as target_file:
            prediction_list = [get_tag(prediction) for prediction in prediction_list]
            target_list = [get_tag(target) for target in target_list]

            raw_target_from_file = target_file.read().split('\n')
            raw_target_from_file = [tar.split() if tar else ' ' for tar in raw_target_from_file]

            gold_tags = [t[pos_of_tag] if t != ' ' else ' ' for t in raw_target_from_file]
            word = [t[pos_of_word] if t != ' ' else ' ' for t in raw_target_from_file]

            spaces = 0
            for i, target_line in enumerate(gold_tags):

                if gold_tags[i] != ' ':
                    if target_list[i - spaces] != gold_tags[i]:
                        print('!!!!!!!!!  should be'+gold_tags[i]+' but is '+target_list[i - spaces]+'  !!!!!!!!!')
                    result_file.write(word[i] + ' ' +gold_tags[i] + ' ' + prediction_list[i - spaces] + '\n')
                else:
                    result_file.write('\n')
                    spaces += 1