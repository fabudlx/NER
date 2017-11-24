# NER With Neural Networks

End-to-end trainable named entity recognicion

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need a pretrained word embedding model (word2vec, fasttext, ect) in the gensim KeyedVector fashion

* [fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) - big but good, used for the pretrained models in this work
* [GloVe](https://nlp.stanford.edu/projects/glove/) - also good


pip intall zstd, numpy, sklearn, keras, tensorflow

## Examples

The GAN_main file is full of examples.

## Using it

If you only want to use the tagger with a pretraind model from this work, you only need the [fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) model and to adjust the model_folder_path and the path to the pretrained fasttext model:

tag_sentence: tags a list of input sentence and returns the tag sequence

```
w2v_class = EmbeddingModel(r'C:\PATH_TO_W2V_MODEL\wiki.en.vec', lower=True, binary=False)

test_sentence = 'John Wick really liked his dog Chester. They went to New York together to support the Help Orphan Puppies organization!'
tagged_sentence = NER.tag_sentence(w2v_class, test_sentence, 'bi-lstm', 'bi-lstm_connl03_fasttext_eng', 'connl03', 'fasttext_eng')
for word_tag_pair in tagged_sentence:
    print(word_tag_pair)
>>('John', 'B-PER')
>>('Wick', 'I-PER')
>>('really', 'O')
>>('liked', 'O')
>>('his', 'O')
>>('dog', 'O')
>>('Chester', 'B-ORG')
>>('.', 'O')
>>('They', 'O')
>>('went', 'O')
>>('to', 'O')
>>('New', 'B-LOC')
>>('York', 'I-LOC')
>>('together', 'O')
>>('to', 'O')
>>('support', 'O')
>>('the', 'O')
>>('Help', 'O')
>>('Orphan', 'O')
>>('Puppies', 'O')
>>('organization', 'O')
>>('!', 'O')


test_sentence = 'The man went to Hamburg to see his friend Fabian.'
tagged_sentence = NER.tag_sentence(w2v_class, test_sentence, 'bi-lstm', 'bi-lstm_connl03_fasttext_eng', 'connl03', 'fasttext_eng')
for word_tag_pair in tagged_sentence:
    print(word_tag_pair)    
>>('The', 'O')
>>('man', 'O')
>>('went', 'O')
>>('to', 'O')
>>('Hamburg', 'B-LOC')
>>('to', 'O')
>>('see', 'O')
>>('his', 'O')
>>('friend', 'O')
>>('Fabian', 'B-PER')
>>('.', 'O')

```


### Setup for development

Development is ready to use for two datasets:

* coNNL 2003
* germEval 2014

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
