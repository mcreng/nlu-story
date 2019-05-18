import copy
import itertools
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Embedding, LSTM, Dense, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping

root_path = 'gdrive/My Drive/Colab Notebooks/'


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class Dataloader():
    def __init__(self, filename=root_path+'train_stories.csv'):
        """
        Constructing the class would immediately preprocess and load the training data.
        """
        # Group stories together
        df = pd.read_csv(filename)
        df = df.apply(lambda x: x[-5]+' '+x[-4]+' ' +
                      x[-3]+' '+x[-2]+' '+x[-1], axis=1)
        # Tokenize
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df.values.tolist())
        self.vocab = tokenizer.word_index

        # Tokenize again to throw away less frequent words
        tokenizer = Tokenizer(len(self.vocab)-3, oov_token="unk")
        tokenizer.fit_on_texts(df.values.tolist())
        with open(root_path+'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

        # Tokenize every stories
        self.vocab = tokenizer.word_index
        self.x = tokenizer.texts_to_sequences(df)
        self.maxlen = max([len(seq) for seq in self.x])
        self.n = len(self.x)

        # Padding and prepare for training
        self.x = pad_sequences(self.x, maxlen=self.maxlen)
        self.y = copy.deepcopy(self.x)
        self.x = list(map(lambda a: a[:-1], self.x))
        self.y = list(map(lambda a: list(map(lambda b: [b], a[1:])), self.y))
        self.x = np.vstack(self.x)
        self.y = np.array(self.y)
