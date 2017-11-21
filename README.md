# NER With Neural Networks

End-to-end trainable named entity recognicion

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need a pretrained word embedding model (word2vec, fasttext, ect) in the gensim KeyedVector fashion

* [fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) - big but good
* [GloVe](https://nlp.stanford.edu/projects/glove/) - also good


pip intall zstd, numpy, sklearn, keras, tensorflow


### Setup for development

Development is ready to use for two datasets:
*coNNL 2003
*germEval 2014

Download the whole thing. 
Change links in 'NER' class: 

raw_data_path: data in text form

data_folder_path: saves vector representation of data (creates subfolder with specific dataset)

model_folder_path: saves models (creates subfolder with specific dataset)

All datasets need to end in either .train, .dev or .test!


```
raw_data_path = r'C:\Users\Me\data\NER\Datensätze\\'
model_folder_path = r'C:\Users\Me\PycharmProjects\NER\Resources\NN_Models\\'
data_folder_path = r'C:\Users\Me\PycharmProjects\NER\Resources\Data\\'
```

## Development

'NER' class offers all the needed functions:

```
create_training_data: creates a vector representation of your training, dev and test data and compresses it and saves it to data_folder_path
train_model: trains model and saves it to model_folder_path
test_model: tests model and saves results to disk
```

## Examples
Can be found in 'main' class

create_training_data:
```
w2v_class = EmbeddingModel(r'path_to_pretrained_model\pretrained Model\wiki.de.vec', lower=False, binary=True)
#lower for models that only know lower case words, binary for binary pretrained w2v model or textform
NER.create_training_data(w2v_class=w2v_class,embedding_model='fasttext_deu',data_set='connl03',language='deu',pos_of_tag=4)
```

training model:
```
NER.train_model(model_type='bi-lstm', model_name='bi-lstm_connl03_fasttext_deu', batch_size=300, epochs=5,data_set='connl03',embedding_model='fasttext_deu')
#model_type: bi-lstm, bi-lstm_crf, bi-lstm_char, bi-lstm_crf_char
#model_name for reloading model
```

testing model:
```
NER.test_model(model_type='bi-lstm', model_name='bi-lstm_connl03_fasttext_deu', test_set_ending='test',data_set='connl03', embedding_model='fasttext_deu', pos_of_tag=4)
```

## Using it

If you only want to use the tagger, you only need the word 2 vec model and to adjust the model_folder_path:

tag_sentence: tags a list of input sentence and returns the tag sequence

```
w2v_class = EmbeddingModel(r'path_to_pretrained_model\pretrained Model\wiki.de.vec', lower=False, binary=True)
list_of_sentences = ['How nice it is to live in Hamburg']
NER.tag_sentence(w2v_class, model_type='bi-lstm', model_name='bi-lstm_connl03_fasttext_deu',data_set='connl03', embedding_model='fasttext_deu', model, list_of_sentences)

```

