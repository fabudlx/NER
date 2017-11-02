from Model.WordEmbeddings import WordEmbedding



def evaluate_word2vec_model(file = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_eng_IBO2.train', word_loc = 0, w2v_path = r'C:\Users\fkarl\Desktop\Science Stuff\pretrained Model\\',w2v_model_name = 'wiki.en.vec', lower = True, binary=True ):
    w2v_model = WordEmbedding.EmbeddingModel(path=w2v_path+w2v_model_name, binary=binary)

    raw_data = open(file, 'r').read().split('\n')
    raw_data = [word_line.split(' ') for word_line in raw_data]

    all_words = [line[word_loc] for line in raw_data if line]

    return_list, not_known_words = w2v_model.get_embedding_improved(all_words, True)


    with open(r'C:\Users\fkarl\PycharmProjects\NER\Model\WordEmbeddings\eval\w2v_eval', 'a') as eval_file:
        eval_file.write('Evaluation of Embedding Model: '+w2v_model_name+'\n')
        # eval_file.write('Unknown words:'+' '.join(not_known_words)+'\n')
        acc = len(not_known_words)/len(return_list)
        eval_file.write('Accuracy:'+str(acc)+'\n')
        eval_file.write('__________________________________________________\n\n\n')


evaluate_word2vec_model(file = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_eng_IBO2.train', w2v_model_name = 'en.wiki.bpe.op200000.d300.w2v.bin',binary=True)
evaluate_word2vec_model(file = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_eng_IBO2.train', w2v_model_name = 'wiki.en.vec',binary=False)


evaluate_word2vec_model(file = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_deu_IBO2.train', w2v_model_name = 'de.wiki.bpe.op200000.d300.w2v.bin',binary=True)
evaluate_word2vec_model(file = r'C:\Users\fkarl\Desktop\Science Stuff\NER\Datensätze\connl03\ner_deu_IBO2.train', w2v_model_name = 'wiki.de.vec',binary=False)

# model loaded
# Counter({True: 28653, False: 18013})
# accuracy: 0.3859983714053058
# Total: 46666




