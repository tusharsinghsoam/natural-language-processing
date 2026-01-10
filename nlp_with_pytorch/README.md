# About

This repository contains a collection of Natural Language Processing (NLP) projects built with PyTorch. The projects demonstrate various NLP techniques and architectures, from foundational concepts to advanced implementations.


## Character-Level RNN: Classifying Names

- We will be building and training a basic character-level Recurrent Neural Network (RNN) to classify words.

- A character-level RNN reads words as a series of characters - outputting a prediction and “hidden state” at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to.

- We’ll train on a few thousand surnames from 18 languages of origin, and predict which language a name is from based on the spelling.

## Character-Level RNN: Generating Names

- We will be building and training a character-level Recurrent Neural Network (RNN) to generate new names based on language origin.

- This generative RNN takes a language category and starting letter as input, then generates plausible names character-by-character by predicting the next letter at each step based on the previous characters and hidden state.

- We train the model by teaching it to predict the next character in real names from 18 different languages, using an end-of-sequence (EOS) marker to indicate when the name is complete.

- After training, we can sample creative new names by providing a language category and starting letter, and the model generates realistic-sounding names letter by letter until it decides the name is complete.


## Acknowledgment

1. [NLP From Scratch: Classifying Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) tutorial.
2. [NLP From Scratch: Generating Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) tutorial.


