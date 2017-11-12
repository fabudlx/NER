from Model import NER
from Model.WordEmbeddings.WordEmbedding import EmbeddingModel

# NER.train_model('bi-lstm_crf','new_model')
# NER.test_model('bi-lstm_crf','bi-lstm_crf_connl03_fasttext_IBO2_en')


# w2v_class = EmbeddingModel(r'C:\Users\fkarl\Desktop\Science Stuff\pretrained Model\german.model',lower=False, binary=True)
#
# NER.create_training_data(w2v_class,'german.model','sharedTask',language='deu',pos_of_tag=2)
# NER.create_training_data(w2v_class,'german.model','connl03',language='deu',pos_of_tag=4)

# w2v_class = EmbeddingModel(r'C:\Users\fkarl\Desktop\Science Stuff\pretrained Model\wiki.de.vec',lower=True, binary=False)

# NER.create_training_data(w2v_class,'fasttext_deu','sharedTask',language='deu',pos_of_tag=2)
# NER.create_training_data(w2v_class,'fasttext_deu','connl03',language='deu',pos_of_tag=4)
#
#
# w2v_class = EmbeddingModel(r'C:\Users\fkarl\Desktop\Science Stuff\pretrained Model\wiki.en.vec',lower=True, binary=False)
#
# NER.create_training_data(w2v_class,'fasttext_eng','connl03',language='eng',pos_of_tag=3)

# NER.train_model('bi-lstm_char', 'bi-lstm_crf_char_sharedTask_german.model',300,10,'connl03','german.model')
# NER.test_model('bi-lstm_crf_char','bi-lstm_crf_char_sharedTask_german.model','dev','sharedTask','german.model',data_format='',language='',pos_of_tag=2)


# NER.train_model('bi-lstm', 'bi-lstm_connl03_fasttext_eng',512,15,'connl03','fasttext_eng')

# NER.train_model('bi-lstm_crf', 'bi-lstm_crf_connl03_fasttext_eng',300,15,'connl03','fasttext_eng')

# NER.train_model('bi-lstm_char', 'bi-lstm_char_connl03_fasttext_eng', 512, 15,'connl03','fasttext_eng')

# NER.train_model('bi-lstm_crf_char', 'bi-lstm_crf_char_connl03_fasttext_eng',300,15,'connl03','fasttext_eng')



NER.train_model('bi-lstm_char', 'bi-lstm_char_connl03_fasttext_deu',512, 15,'connl03','fasttext_deu')
NER.train_model('bi-lstm_char', 'bi-lstm_char_sharedTask_fasttext_deu',512, 15,'sharedTask','fasttext_deu')

# NER.train_model('bi-lstm_crf_char', 'bi-lstm_crf_char_connl03_fasttext_deu',300, 15,'connl03','fasttext_deu')
# NER.train_model('bi-lstm_crf_char', 'bi-lstm_crf_char_sharedTask_fasttext_deu',512, 15,'sharedTask','fasttext_deu')


# NER.test_model('bi-lstm_char','bi-lstm_char_connl03_fasttext_deu','dev','connl03','fasttext_deu',data_format='IBO2',language='deu',pos_of_tag=4)
# NER.test_model('bi-lstm_char','bi-lstm_char_sharedTask_fasttext_deu','dev','connl03','fasttext_deu',data_format='IBO2',language='deu',pos_of_tag=4)

# NER.test_model('bi-lstm_crf_char','bi-lstm_crf_char_connl03_fasttext_deu','dev','connl03','fasttext_deu',data_format='IBO2',language='deu',pos_of_tag=4)
# NER.test_model('bi-lstm_crf_char','bi-lstm_crf_char_sharedTask_fasttext_deu','dev','connl03','fasttext_deu',data_format='IBO2',language='deu',pos_of_tag=4)

# NER.test_model('bi-lstm','bi-lstm_connl03_fasttext_eng','dev','connl03','fasttext_eng',data_format='IBO2',language='eng',pos_of_tag=3)
# NER.test_model('bi-lstm_crf','bi-lstm_crf_connl03_fasttext_eng','dev','connl03','fasttext_eng',data_format='IBO2',language='eng',pos_of_tag=3)
# NER.test_model('bi-lstm_char','bi-lstm_char_connl03_fasttext_eng','dev','connl03','fasttext_eng',data_format='IBO2',language='eng',pos_of_tag=3)
# NER.test_model('bi-lstm_crf_char','bi-lstm_crf_char_connl03_fasttext_eng','dev','connl03','fasttext_eng',data_format='IBO2',language='eng',pos_of_tag=3)