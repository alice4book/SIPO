import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from knn import KnnModel
from text_preprocessor import TextPreprocessor
from sentiment_model import SentimentModel
import pickle

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis")

        self.model_type = None
        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()

        self.setup_ui()

    def setup_ui(self):
        self.model_label = tk.Label(self.root, text="Choose Model:")
        self.model_label.pack()

        self.knn_button = tk.Button(self.root, text="KNN", command=lambda: self.choose_model("KNN"))
        self.knn_button.pack()

        self.cnn_button = tk.Button(self.root, text="CNN", command=lambda: self.choose_model("CNN"))
        self.cnn_button.pack()

        self.load_model_button = tk.Button(self.root, text="Load Model", command=self.load_model)
        self.load_model_button.pack()

        self.train_model_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_model_button.pack()

        self.text_label = tk.Label(self.root, text="Enter your review:")
        self.text_label.pack()

        self.text_entry = tk.Text(self.root, height=10, width=50)
        self.text_entry.pack()

        self.clear_button = tk.Button(self.root, text="Clear Text", command=self.clear_text)
        self.clear_button.pack()

        self.predict_button = tk.Button(self.root, text="Predict Sentiment", command=self.predict_sentiment)
        self.predict_button.pack()

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit)
        self.exit_button.pack()

    def choose_model(self, model_type):
        self.model_type = model_type
        messagebox.showinfo("Model Chosen", f"{model_type} model chosen.")

    def load_model(self):
        if not self.model_type:
            messagebox.showerror("Error", "Please choose a model first.")
            return

        model_path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5"), ("Pickle files", "*.pkl"), ("Joblib", "*.joblib")])
        if not model_path:
            return

        if self.model_type == "CNN":
            self.model = SentimentModel()
            self.model.load_model(model_path)
        elif self.model_type == "KNN":
            try:
                self.vectorizer = self.load_vectorizer('vectorizer.pkl')
                self.model = KnnModel()
                self.model.load(model_path)
            except FileNotFoundError:
                messagebox.showerror("Error", "Vectorizer file not found. Train the model first.")

        messagebox.showinfo("Model Loaded", f"{self.model_type} model loaded from {model_path}.")

    def train_model(self):
        if not self.model_type:
            messagebox.showerror("Error", "Please choose a model first.")
            return

        data_loader = DataLoader(dataset_path='DatasetPL.csv')
        reviews, sentiments = data_loader.load_data()

        translated_reviews = [review for review in reviews]

        X, y = self.preprocessor.prepare_data(translated_reviews, sentiments)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_type == "KNN":
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X_knn = self.vectorizer.fit_transform(translated_reviews)
            X_train_knn, X_val_knn, y_train_knn, y_val_knn = train_test_split(X_knn, y, test_size=0.2, random_state=42)
            self.model = KnnModel()
            self.model.fit(X_train_knn, y_train_knn)
            self.report(X_val_knn, y_val_knn)
            self.save_vectorizer(self.vectorizer, 'vectorizer.pkl')
            self.model.save('knn_model.joblib')
        elif self.model_type == "CNN":
            self.model = SentimentModel()
            self.model.compile_model()
            self.model.train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
            self.model.save_model('sentiment_model.h5')

        messagebox.showinfo("Training Complete", f"{self.model_type} model training complete and saved.")

    def predict_sentiment(self):
        if not self.model:
            messagebox.showerror("Error", "Please load or train a model first.")
            return

        text = self.text_entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter a review text.")
            return

        if self.model_type == "CNN":
            prediction = self.model.predict(text, self.preprocessor)
            sentiment = "positive" if prediction > 0.5 else "negative"
        elif self.model_type == "KNN":
            processed_text = self.vectorizer.transform([text])
            prediction = self.model.predict(text ,self.vectorizer)
            sentiment = "positive" if prediction > 0.5 else "negative"

        self.result_label.config(text=f"Sentiment: {sentiment}")

    def clear_text(self):
        self.text_entry.delete("1.0", tk.END)

    def load_vectorizer(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_vectorizer(self, vectorizer, path):
        with open(path, 'wb') as f:
            pickle.dump(vectorizer, f)

    def report(self, X_val, y_val):
        if self.model_type == "KNN":
            self.model.report(X_val, y_val)

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
