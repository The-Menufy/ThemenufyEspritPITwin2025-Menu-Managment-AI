import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import nltk
nltk.download('wordnet', quiet=True)  # Avoid verbose download message
from nltk.stem import WordNetLemmatizer

class ClusteringModel:
    def __init__(self, dataset_path):
        print("[ClusteringModel] Loading and cleaning dataset...")
        self.df = pd.read_csv(dataset_path)

        # Clean text
        self.df['Dish Name'] = self.df['Dish Name'].astype(str).str.lower().str.strip()
        self.df['Ingredients'] = self.df['Ingredients'].astype(str).str.lower().str.strip()

        # Remove rows with missing values
        self.df = self.df[self.df['Ingredients'].notnull() & self.df['Dish Name'].notnull()]

        # Remove duplicates
        self.df.drop_duplicates(subset=['Dish Name', 'Ingredients'], inplace=True)

        # Lemmatize ingredients
        lemmatizer = WordNetLemmatizer()
        self.df['Ingredients'] = self.df['Ingredients'].apply(
            lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
        )


        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
        X = self.vectorizer.fit_transform(self.df['Ingredients'])

        # KMeans Clustering (more clusters to improve specificity)
        self.kmeans = KMeans(n_clusters=7, random_state=42, n_init=20)
        self.df['Cluster'] = self.kmeans.fit_predict(X)

        # Custom category labels per cluster (update as needed after WordCloud analysis)
        self.cluster_labels = {
            0: 'Salads & Wraps',
            1: 'Desserts & Pastries',
            2: 'Curries & Stews',
            3: 'Baked Dishes',
            4: 'Asian & Stir Fry',
            5: 'Breakfast & Brunch',
            6: 'Grilled & Roasted'
        }
        self.df['Cluster_Label'] = self.df['Cluster'].map(self.cluster_labels)

        # Save models
        joblib.dump(self.kmeans, 'models/clustering_model.pkl')
        joblib.dump(self.vectorizer, 'models/clustering_vectorizer.pkl')
        self.df.to_csv('data/clustered_dishes.csv', index=False)

        print("[ClusteringModel] Model trained and saved ✅")

    def predict(self, dish_name):
        if not dish_name:
            return {'cluster': -1, 'category': 'Unknown'}

        dish_name = dish_name.strip().lower()
        match = self.df[self.df['Dish Name'] == dish_name]

        if not match.empty:
            row = match.iloc[0]
            return {
                'cluster': int(row['Cluster']),
                'category': row['Cluster_Label']
            }

        # Predict using model
        vectorizer = joblib.load('models/clustering_vectorizer.pkl')
        model = joblib.load('models/clustering_model.pkl')
        X = vectorizer.transform([dish_name])
        cluster = int(model.predict(X)[0])
        category = self.cluster_labels.get(cluster, 'Unknown')
        return {'cluster': cluster, 'category': category}
