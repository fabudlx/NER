from Model import SaveAndLoad as sl
from sklearn import preprocessing

char_embedding_train = sl.load_padded_character_embedding_list(path=r'C:\Users\fkarl\PycharmProjects\NER\Resources\Data\connl03\fasttext_eng', variable_name='train')


normalized = preprocessing.normalize(char_embedding_train)
print(normalized[:10])