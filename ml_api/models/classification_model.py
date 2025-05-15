import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationModel:
    def __init__(self, dataset_path="data/Generated_Recipe_Dataset.csv"):
        # Load the dataset
        self.df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {self.df.shape[0]} rows")

        # Ensure required columns exist
        required_columns = ['Dish Name', 'Ingredients']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            logger.error(f"Missing columns in dataset: {missing_columns}")
            raise ValueError(f"Dataset missing columns: {missing_columns}")

        # Encode dish names
        self.label_encoder = LabelEncoder()
        self.df['Dish Name'] = self.label_encoder.fit_transform(self.df['Dish Name'])

        # Data Augmentation: Add Stronger Noise (More Randomness)
        def add_noise(text, remove_prob=0.50, swap_prob=0.40):
            words = text.split()
            new_words = []
            # Randomly remove words
            for word in words:
                if random.random() < remove_prob:
                    continue
                new_words.append(word)
            # Swap random words
            if len(new_words) > 1:
                for _ in range(int(len(new_words) * swap_prob)):
                    idx1, idx2 = random.sample(range(len(new_words)), 2)
                    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            return " ".join(new_words)

        self.df["Ingredients"] = self.df["Ingredients"].apply(add_noise)

        # Vectorize ingredients using TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=100,
            ngram_range=(1, 2)
        )
        X = self.vectorizer.fit_transform(self.df['Ingredients'])
        y = self.df['Dish Name']

        # Split data (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # Drop 60% of training samples to reduce overfitting
        train_indices = np.random.choice(len(X_train.toarray()), int(0.4 * len(X_train.toarray())), replace=False)
        X_train = X_train[train_indices]
        y_train = np.array(y_train)[train_indices]

        # Train KNN
        self.knn = KNeighborsClassifier(n_neighbors=20, weights='distance', metric='cosine')
        self.knn.fit(X_train, y_train)
        y_pred_knn = self.knn.predict(X_test)
        self.knn_accuracy = accuracy_score(y_test, y_pred_knn)
        logger.info(f"KNN Accuracy: {self.knn_accuracy:.2f}")
        print(f"\n🎯 KNN Accuracy: {self.knn_accuracy:.2f}")
        print("📄 KNN Classification Report:\n", classification_report(y_test, y_pred_knn, zero_division=0))

        # Train SVM
        self.svm = SVC(kernel='linear', C=0.05, class_weight='balanced')
        self.svm.fit(X_train, y_train)
        y_pred_svm = self.svm.predict(X_test)
        self.svm_accuracy = accuracy_score(y_test, y_pred_svm)
        logger.info(f"SVM Accuracy: {self.svm_accuracy:.2f}")
        print(f"\n🎯 SVM Accuracy: {self.svm_accuracy:.2f}")
        print("📄 SVM Classification Report:\n", classification_report(y_test, y_pred_svm, zero_division=0))

    def predict(self, ingredients_text):
        try:
            ingredients_list = [word.strip() for word in ingredients_text.replace('/', ',').split(',') if word.strip()]
            if not ingredients_list:
                logger.error("No valid ingredients provided")
                return {"KNN Prediction": "N/A", "SVM Prediction": "N/A", "KNN Accuracy": 0.0, "SVM Accuracy": 0.0}
            ingredients_text = " ".join(ingredients_list)  # Format as "milk"
            logger.info(f"Processing input: {ingredients_text}")

            ingredients_vector = self.vectorizer.transform([ingredients_text])
            if ingredients_vector.nnz == 0:
                logger.warning("Input vectorized to zero features")
                return {"KNN Prediction": "N/A", "SVM Prediction": "N/A", "KNN Accuracy": 0.0, "SVM Accuracy": 0.0}

            knn_prediction = self.label_encoder.inverse_transform(self.knn.predict(ingredients_vector))[0]
            svm_prediction = self.label_encoder.inverse_transform(self.svm.predict(ingredients_vector))[0]
            logger.info(f"KNN Prediction: {knn_prediction}, SVM Prediction: {svm_prediction}")

            return {
                "KNN Prediction": knn_prediction,
                "SVM Prediction": svm_prediction,
                "KNN Accuracy": self.knn_accuracy,
                "SVM Accuracy": self.svm_accuracy
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"KNN Prediction": "N/A", "SVM Prediction": "N/A", "KNN Accuracy": 0.0, "SVM Accuracy": 0.0}