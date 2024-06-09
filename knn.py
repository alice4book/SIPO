import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_loader import DataLoader

class KnnModel:
    """
        # Make predictions on the test set
        y_pred = knn.predict(X_val)

        # Evaluate the classifier
        accuracy = accuracy_score(y_val, y_pred)

        report = classification_report(y_val, y_pred, target_names=['negative', 'positive'])

        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)

        while True:
            text = input("Wprowadź recenzję do oceny (lub wpisz 'exit', aby zakończyć): ")
            if text.lower() == 'exit':
                break
            value = vectorizer.transform([text])
            prediction = knn.predict(value)
            sentiment = "positive" if prediction > 0.5 else "negative"
            print(f"Ocena recenzji: {sentiment}")
    """



    def __init__(self, n=4):
        self.knn_model = KNeighborsClassifier(n_neighbors=n)


    def fit(self, X_train, y_train):
        self.knn_model.fit(X_train, y_train)


    def predict(self, text, vectorizer):
        value = vectorizer.transform([text])
        prediction = self.knn_model.predict(value)
        return prediction


    def report(self, X_val, y_val):
        # Make predictions on the test set
        y_pred = self.knn_model.predict(X_val)

        # Evaluate the classifier
        accuracy = accuracy_score(y_val, y_pred)

        report = classification_report(y_val, y_pred, target_names=['negative', 'positive'])

        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)