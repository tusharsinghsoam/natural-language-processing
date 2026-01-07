# About

## Basic Text Classification

1. This notebook trains a sentiment analysis model to classify movie reviews as positive or negative, based on the text of the review. This is an example of binary—or two-class—classification, an important and widely applicable kind of machine learning problem.

2. We'll use the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) that contains the text of 50,000 movie reviews from the [Internet Movie Database](https://www.imdb.com/). 

3. These are split into 25,000 reviews for training and 25,000 reviews for testing. The training and testing sets are *balanced*, meaning they contain an equal number of positive and negative reviews.

## Handling Text data in Tensorflow

* This project demonstrates two ways to load and preprocess text.

    - First, you will use Keras utilities and preprocessing layers. These include `tf.keras.utils.text_dataset_from_directory` to turn data into a `tf.data.Dataset` and `tf.keras.layers.TextVectorization` for data standardization, tokenization, and vectorization. If you are new to TensorFlow, you should start with these.
    - Then, you will use lower-level utilities like `tf.data.TextLineDataset` to load text files, `tf.lookup` for custom in-model lookup tables, and [TensorFlow Text](https://www.tensorflow.org/text) APIs, such as `text.UnicodeScriptTokenizer` and `text.case_fold_utf8`, to preprocess the data for finer-grain control.
    - We will also learn to download many more datasets from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/overview). We will use the [IMDB Large Movie Review dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) to train a model for sentiment classification.

## Text classification with an RNN

1. This text classification project trains a [recurrent neural network](https://developers.google.com/machine-learning/glossary/#recurrent_neural_network) on the [IMDB large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) for sentiment analysis.

2. The IMDB large movie review dataset is a *binary classification* dataset—all the reviews have either a *positive* or *negative* sentiment.

3. We will download the dataset using [TFDS](https://www.tensorflow.org/datasets). See the [loading text project](https://www.tensorflow.org/tutorials/load_data/text) for details on how to load this sort of data manually.

## Text classification with BERT

1. This text classification contains complete code to fine-tune BERT to perform sentiment analysis on a dataset of plain-text IMDB movie reviews. In addition to training a model, you will learn how to preprocess text into an appropriate format.

2. We'll use the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) that contains the text of 50,000 movie reviews from the [Internet Movie Database](https://www.imdb.com/).

3. [BERT](https://arxiv.org/abs/1810.04805) and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). They compute vector-space representations of natural language that are suitable for use in deep learning models. The BERT family of models uses the Transformer encoder architecture to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers.


## Acknowledgment

1. Beginner Tutorial named as [Basic Text Classification](https://www.tensorflow.org/tutorials/keras/text_classification#exercise_multi-class_classification_on_stack_overflow_questions) at tensorflow website.
2. [Load Text](https://www.tensorflow.org/tutorials/load_data/text) tutorial.
3. [Text classification with an RNN](https://www.tensorflow.org/text/tutorials/text_classification_rnn) tutorial.
4. [Text classification with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert) tutorial.
