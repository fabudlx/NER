# NER With Neural Networks

End-to-end trainable named entity recognicion

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need a pretrained word embedding model (word2vec, fasttext, ect) in the gensim KeyedVector fashion

* [fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) - big but good
* [GloVe](https://nlp.stanford.edu/projects/glove/) - also good


pip intall zstd, numpy, sklearn, keras, tensorflow


### Setup

Download the whole thing. 
Change links in NER: 

raw_data_path: data in text form

data_folder_path: saves vector representation of data (creates subfolder with specific dataset)

model_folder_path: saves models (creates subfolder with specific dataset)


```
raw_data_path = r'C:\Users\Me\data\NER\Datensätze\\'
model_folder_path = r'C:\Users\Me\PycharmProjects\NER\Resources\NN_Models\\'
data_folder_path = r'C:\Users\Me\PycharmProjects\NER\Resources\Data\\'
```

## Running

NER class offers all the needed functions:

create_training_data: creates a vector representation of your training, dev and test data and compresses it and saves it to data_folder_path
```
train_model: trains model and saves it to model_folder_path
test_model: tests model and saves results to disk
tag_sentence: tags a list of input sentence and returns the tag sequence
```
