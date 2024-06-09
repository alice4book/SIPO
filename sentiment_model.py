from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class SentimentModel:
    def __init__(self, max_words=10000, max_len=200, embedding_dim=50):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=self.embedding_dim, input_length=self.max_len))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def compile_model(self):
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        from tensorflow.keras.models import load_model
        self.model = load_model(path)

    def predict(self, text, preprocessor):
        processed_text = preprocessor.transform([text])
        prediction = self.model.predict(processed_text)
        return prediction[0][0]