import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from knn import KnnModel
from text_preprocessor import TextPreprocessor
from sentiment_model import SentimentModel
from translator import TranslatorModule


def main():
    # Wczytywanie danych
    data_loader = DataLoader(dataset_path='IMDB Dataset.csv')
    reviews, sentiments = data_loader.load_data()

    translator = TranslatorModule()
    startTime = time.time()
    translated_reviews = []
    for review in reviews:
        print(review)
        time.sleep(1)
        translated_review = translator.translate(review)
        translated_reviews.append(translated_review)

    # Przetwarzanie danych
    preprocessor = TextPreprocessor()
    X, y = preprocessor.prepare_data(translated_reviews, sentiments)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)

    # Knn setup
    X_knn = vectorizer.fit_transform(translated_reviews)
    X_train_knn, X_val_knn, y_train_knn, y_val_knn = train_test_split(X_knn, y, test_size=0.2, random_state=42)
    knn_model = KnnModel()
    knn_model.fit(X_train_knn, y_train_knn)

    # Konwersja do numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Budowanie i trenowanie modelu
    model = SentimentModel()
    model.compile_model()
    model.train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

    # Zapisywanie modelu
    model.save_model('sentiment_model.h5')

    # Klasyfikacja nowej recenzji
    while True:
        text = input("Wprowadź recenzję do oceny (lub wpisz 'exit', aby zakończyć): ")
        if text.lower() == 'exit':
            break

        prediction = model.predict(text, preprocessor)
        knn_prediction = knn_model.predict(text, vectorizer)
        sentiment = "positive" if prediction > 0.5 else "negative"
        sentiment_knn = "positive" if knn_prediction > 0.5 else "negative"
        print(f"Ocena recenzji CNN: {sentiment}")
        print(f"Ocena recenzji KNN: {sentiment_knn}")


if __name__ == "__main__":
    main()