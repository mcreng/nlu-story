import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class LMEvaluator():
    def __init__(self, **filenames):
        """
        Load all required files for evaluation.
        """
        with open(filenames['tokenizer'], 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(filenames['dataloader'], 'rb') as f:
            self.dataloader = pickle.load(f)
        self.model = load_model(filenames['model'])

    def evaluate(self, sentences):
        """
        Evaluate probabilities of provided sentences.

        Arguments:
            sentences (array[str]): Array of sentences

        Returns:
            np.ndarray: Array of probabilities
        """
        x_test = self.tokenizer.texts_to_sequences(sentences)
        n_tok = list(map(len, x_test))
        y_test = x_test
        x_test = pad_sequences(x_test, maxlen=self.dataloader.maxlen)
        x_test = x_test[:, :-1]
        x_test = np.array(x_test)
        p_pred = self.model.predict(x_test)

        log_probs = []
        for idx, p in enumerate(p_pred):
            p = p[-n_tok[idx]:, :]
            p = p[range(n_tok[idx]), y_test[idx]]
            log_prob = np.sum(np.log(p))
            log_probs.append(log_prob)
        return log_probs
