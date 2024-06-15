import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_loader import DataLoader

class KnnModel:
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

    def save(self, filename):
        # Save the trained model to a file
        joblib.dump(self.knn_model, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        # Load a trained model from a file
        self.knn_model = joblib.load(filename)
        print(f"Model loaded from {filename}")