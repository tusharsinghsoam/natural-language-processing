# About

This repository contains a collection of Natural Language Processing (NLP) projects built with PyTorch. The projects demonstrate various NLP techniques and architectures, from foundational concepts to advanced implementations.


### Character-Level RNN
- We will be building and training a basic character-level Recurrent Neural Network (RNN) to classify words.

- A character-level RNN reads words as a series of characters - outputting a prediction and “hidden state” at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to.

- We’ll train on a few thousand surnames from 18 languages of origin, and predict which language a name is from based on the spelling.

## Acknowledgment

1. [NLP From Scratch: Classifying Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) tutorial.


