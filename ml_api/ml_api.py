from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import traceback

from models.classification_model import ClassificationModel
from models.recommendation_model import RecommendationModel
from models.regression_model import RegressionModel
from models.clustering_model import ClusteringModel

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Initialize models
try:
    classification_model = ClassificationModel(dataset_path=os.path.join(DATA_DIR, 'Generated_Recipe_Dataset.csv'))
    recommendation_model = RecommendationModel(dataset_path=os.path.join(DATA_DIR, 'World_Dishes_Dataset (1).csv'))
    regression_model = RegressionModel(dataset_path=os.path.join(DATA_DIR, 'World_Dishes_Dataset (1).csv'))
    clustering_model = ClusteringModel(dataset_path=os.path.join(DATA_DIR, 'Generated_Recipe_Dataset.csv'))
except FileNotFoundError as e:
    print(f"❌ Dataset not found: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error initializing models: {e}")
    traceback.print_exc()
    exit(1)

# ==============================
# CLASSIFICATION
# ==============================
@app.route('/api/classification', methods=['POST'])
def classification():
    try:
        data = request.get_json()
        ingredients = data.get('ingredients', '')
        if not ingredients:
            return jsonify({'error': 'No ingredients provided'}), 400

        prediction = classification_model.predict(ingredients)
        return jsonify(prediction)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

# ==============================
# RECOMMENDATION
# ==============================
@app.route('/api/recommendation', methods=['POST'])
def recommendation():
    try:
        data = request.get_json()
        input_ingredients = data.get('ingredients', [])
        if not input_ingredients:
            return jsonify({'error': 'No ingredients provided'}), 400

        result = recommendation_model.recommend(input_ingredients)
        for dish in result:
            dish_details = recommendation_model.df.loc[
                recommendation_model.df['Dish Name'] == dish['Dish Name'], [
                    'Dish Name', 'Ingredients', 'Num_Ingredients', 'Average_Ingredient_Calorie',
                    'Preparation_Time_Minutes', 'Cooking_Time_Minutes', 'Calories', 'Category', 'Instructions'
                ]
            ].iloc[0].to_dict()
            dish.update(dish_details)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Recommendation failed: {str(e)}'}), 500

# ==============================
# CLUSTERING
# ==============================
@app.route('/api/clustering', methods=['POST'])
def clustering():
    try:
        data = request.get_json()
        dish_name = data.get('dish_name', '')
        if not dish_name:
            return jsonify({'error': 'No dish name provided'}), 400

        result = clustering_model.predict(dish_name)
        return jsonify({
            'dish_name': dish_name,
            'category': result['category'],
            'cluster': result['cluster']
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Clustering failed: {str(e)}'}), 500

# ==============================
# REGRESSION
# ==============================
@app.route('/api/regression', methods=['POST'])
def regression():
    try:
        data = request.get_json()
        input_features = data.get('features')
        avg_cal_per_ingredient = data.get('avg_cal_per_ingredient', 50)

        if not input_features or len(input_features) != 3 or not all(isinstance(x, (int, float)) for x in input_features):
            return jsonify({
                'error': 'Invalid input: Provide 3 numeric values [Num_Ingredients, Preparation_Time_Minutes, Cooking_Time_Minutes]'
            }), 400

        if any(x < 0 for x in input_features) or avg_cal_per_ingredient < 0:
            return jsonify({'error': 'All input values must be non-negative'}), 400

        predicted_calories = regression_model.predict(input_features + [avg_cal_per_ingredient])
        if predicted_calories is None:
            return jsonify({
                'error': 'Failed to predict calories',
                'predicted_calories': None,
                'matching_dishes': []
            }), 500

        tolerance = 50
        min_cal = max(0, predicted_calories - tolerance)
        max_cal = predicted_calories + tolerance

        regression_model.df['Calories'] = pd.to_numeric(regression_model.df['Calories'], errors='coerce')
        matching_dishes = regression_model.df[
            regression_model.df['Calories'].between(min_cal, max_cal)
        ][[
            'Dish Name', 'Ingredients', 'Num_Ingredients', 'Average_Ingredient_Calorie',
            'Preparation_Time_Minutes', 'Cooking_Time_Minutes', 'Category', 'Calories', 'Instructions'
        ]].dropna().to_dict('records')

        return jsonify({
            'predicted_calories': predicted_calories,
            'matching_dishes': matching_dishes[:5]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Regression failed: {str(e)}'}), 500

# ==============================
# Run app (without debug message)
# ==============================
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)
