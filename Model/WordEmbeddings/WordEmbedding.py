from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import re


EMBEDDING_SIZE = 300

class EmbeddingModel():

    def __init__(self, path = r'C:\Users\fkarl\Desktop\Science Stuff\pretrained Model\wiki.en.vec', binary = False):
        print('loading w2v model... this could take a while')
        self.w2v_model = KeyedVectors.load_word2vec_format(path, binary=binary)
        print('word2vec model loaded')

    def get_embedding_improved(self, list_of_words, lower = False):
        return_list = []
        not_known_words =[]
        for word in list_of_words:
            if lower:
                word = word.lower()
                if word.lower() in self.w2v_model.vocab:
                    return_list.append(self.w2v_model[word.lower()])
                    continue

                elif word.lower() == '-docstart-':
                    return_list.append(np.ones(EMBEDDING_SIZE) - 0.5)
                    continue

                if bool(re.search(r'\d', word)):
                    if '0' in self.w2v_model:
                        return_list.append(self.w2v_model['0'])
                    if 'one' in self.w2v_model:
                        return_list.append(self.w2v_model['one'])
                    else:
                        return_list.append(np.ones(EMBEDDING_SIZE) - 0.8)
                    continue

                if '-' in word:
                    words = re.split('-', word)

                    for w in words:
                        if w.lower() in self.w2v_model.vocab:
                            return_list.append(self.w2v_model[w.lower()])
                        continue
                        word_without_punctuation = re.sub(r'[^\w\s]', '', w)
                        if word_without_punctuation.lower() in self.w2v_model.vocab:
                            return_list.append(self.w2v_model[w.lower()])
                            continue
                        return_list.append(np.ones(EMBEDDING_SIZE) - 2)
                        not_known_words.append(w.lower())
                    continue


                word_without_punctuation = re.sub(r'[^\w\s]', '', word)
                if word_without_punctuation.lower() in self.w2v_model.vocab:
                    return_list.append(self.w2v_model[word_without_punctuation.lower()])
                    continue

                return_list.append(np.ones(EMBEDDING_SIZE) - 2)
                not_known_words.append(word.lower())
            else:
                return [self.w2v_model[y] if y in self.w2v_model.vocab else np.ones(EMBEDDING_SIZE) - 0.5 for y in list_of_words], []

        return return_list, not_known_words

    def get_embedding(self, list_of_words, lower = False):
        if lower:
            return [self.w2v_model[y.lower()] if y in self.w2v_model.vocab else np.ones(EMBEDDING_SIZE) - 0.5 for y in list_of_words]
        else:
            return [self.w2v_model[y] if y in self.w2v_model.vocab else np.ones(EMBEDDING_SIZE) - 0.5 for y in list_of_words]