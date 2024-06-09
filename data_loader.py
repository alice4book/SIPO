import pandas as pd

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        data = pd.read_csv(self.dataset_path)
        reviews = data['review'].tolist()
        sentiments = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).tolist()
        return reviews, sentiments