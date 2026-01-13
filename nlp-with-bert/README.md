# About

- This project explores various use cases and applications of BERT (Bidirectional Encoder Representations from Transformers), a powerful pre-trained language model developed by Google. BERT has revolutionized natural language processing by introducing bidirectional context understanding, making it highly effective for a wide range of NLP tasks.

- BERT uses transformer architecture to understand the context of words in a sentence by looking at both left and right contexts simultaneously. This project demonstrates practical implementations of BERT across different domains and tasks.

## BERT pre-processing with tf-text

- This project will show how to use TF.Text preprocessing ops to transform text data into inputs for the BERT model and inputs for language masking pretraining task described in "Masked LM and Masking Procedure" of [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf).

- This process involves tokenizing text into subword units, combining sentences, trimming content to a fixed size and extracting labels for the masked language modeling task.

## Fine Tuning a BERT Model

- This project uses the GLUE (General Language Understanding Evaluation) MRPC (Microsoft Research Paraphrase Corpus) [dataset from TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/catalog/glue#gluemrpc) to fine-tune a [Bidirectional Encoder Representations from Transformers (BERT)](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018) model using [TensorFlow Model Garden](https://github.com/tensorflow/models).

## Fine Tune a Bert Model: Using Pytorch and Transformer Library

- This project demonstrates fine-tuning the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model for sentence classification using PyTorch and the [Hugging Face Transformers](https://huggingface.co/docs/transformers) library. 

- The implementation uses the [CoLA (Corpus of Linguistic Acceptability)](https://nyu-mll.github.io/CoLA/) dataset to train a binary classifier that determines whether sentences are grammatically acceptable. 

- The model uses BertForSequenceClassification with AdamW optimizer, linear learning rate scheduling, and is evaluated using Matthews Correlation Coefficient (MCC) on out-of-domain test data.

## QA with a fine-tuned BERT

- This project demonstrates question answering using a pre-trained BERT model fine-tuned on the SQuAD (Stanford Question Answering Dataset) dataset. The implementation uses the [bert-large-uncased-whole-word-masking-finetuned-squad](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad) model from Hugging Face.

- The model takes a question and a context passage as input, tokenizes them using BERT's tokenizer, and predicts the start and end positions of the answer span within the context. The project includes visualization of the model's confidence scores for each token position, helping to understand how the model identifies answer boundaries.

- The notebook demonstrates the complete pipeline: tokenization with special tokens ([CLS], [SEP]), model inference to obtain start and end logits, answer extraction by combining subword tokens, and score visualization using matplotlib and seaborn.

## Acknowledgment

1. [BERT Preprocessing with TF Text](https://www.tensorflow.org/text/guide/bert_preprocessing_guide) tutorial.
2. [Fine-tuning a BERT model](https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert)
3. [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
4. [Question Answering with a Fine-Tuned BERT](https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/)



