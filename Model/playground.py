from Model import SaveAndLoad as sl

char_embedding_train = sl.load_padded_character_embedding_list(path=r'C:\Users\fkarl\PycharmProjects\NER\Resources\Data\connl03\fasttext_eng', variable_name='train')


print(char_embedding_train[5:10])