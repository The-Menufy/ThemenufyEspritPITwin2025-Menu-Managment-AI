# models/regression_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class RegressionModel:
    def __init__(self, dataset_path):
        try:
            self.df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        # Define required columns
        required_columns = ['Num_Ingredients', 'Preparation_Time_Minutes', 'Cooking_Time_Minutes', 'Calories', 'Average_Ingredient_Calorie']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Clean and convert columns to numeric
        for col in required_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            if col == 'Calories':
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        # Check for valid data
        if self.df[required_columns].isna().any().any():
            raise ValueError("Dataset contains unhandled NaN values after cleaning")

        self.feature_names = ['Num_Ingredients', 'Preparation_Time_Minutes', 'Cooking_Time_Minutes', 'Average_Ingredient_Calorie']
        print(f"Initialized feature names: {self.feature_names}")
        self.features = self.df[self.feature_names]
        self.target = 'Calories'
        self.model = LinearRegression()
        self.scaler = StandardScaler()

        # Create models directory
        os.makedirs('models', exist_ok=True)

        # Delete existing model files to force retraining
        model_path = 'models/regression_model.pkl'
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Deleted existing model file: {model_path}")
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            print(f"Deleted existing scaler file: {scaler_path}")

        # Train model
        print("Training new model...")
        self.train()

    def train(self):
        try:
            print(f"Training with features: {self.feature_names}")
            print(f"Training data columns: {self.features.columns.tolist()}")
            X_scaled = self.scaler.fit_transform(self.features)
            y = self.df[self.target]

            if y.isna().all():
                raise ValueError("No valid calorie data available for training")

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)

            joblib.dump(self.model, 'models/regression_model.pkl')
            joblib.dump(self.scaler, 'models/scaler.pkl')
            print("Regression model trained and saved successfully")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def predict(self, input_features):
        if not isinstance(input_features, list) or len(input_features) != 4:
            print(f"Invalid input features: {input_features}")
            return None

        # Ensure input features are numeric
        try:
            input_features = [float(x) for x in input_features]
        except (ValueError, TypeError) as e:
            print(f"Error converting input features to float: {e}")
            return None

        # Load model and scaler
        try:
            model = joblib.load('models/regression_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
        except FileNotFoundError as e:
            print(f"Model files not found: {e}. Training a new model...")
            self.train()
            model = joblib.load('models/regression_model.pkl')
            scaler = joblib.load('models/scaler.pkl')

        # Make prediction
        try:
            print(f"Predicting with features: {self.feature_names}")
            input_df = pd.DataFrame([input_features], columns=self.feature_names)
            print(f"Prediction input columns: {input_df.columns.tolist()}")
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            print(f"Raw prediction: {prediction}")
            return float(prediction[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None