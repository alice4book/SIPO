from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class TextPreprocessor:
    def __init__(self, max_words=10000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        return np.array(padded_sequences)

    def prepare_data(self, texts, labels):
        self.fit(texts)
        X = self.transform(texts)
        y = np.array(labels)
        return X, y