# IA

[![License](https://img.shields.io/github/license/The-Menufy/IA.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Issues](https://img.shields.io/github/issues/The-Menufy/IA.svg)](https://github.com/The-Menufy/IA/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/The-Menufy/IA.svg)](https://github.com/The-Menufy/IA/pulls)
[![Last Commit](https://img.shields.io/github/last-commit/The-Menufy/IA.svg)](https://github.com/The-Menufy/IA/commits/main)

---

## 🚀 Project Overview

**IA** is an advanced Python-based machine learning API designed for intelligent food dish classification, recommendation, clustering, and calorie regression. It leverages various ML models and a modular API architecture to provide robust and extensible food data analysis and prediction capabilities.

---

## 📚 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## ✨ Features

- Modular and scalable Python codebase
- Multiple ML models: Classification, Recommendation, Clustering, Regression
- Dishes, ingredients, and calorie prediction APIs
- Dataset-driven, easily extensible with new data
- API powered by Flask, ready for deployment (e.g., with Waitress)
- Pre-trained models and retraining scripts included
- Detailed error handling and logging
- Comprehensive test suite (pytest-ready)
- Easy integration with frontends or other backends

---

## 🏗️ Architecture

The `ml_api/` directory implements a modular machine learning API using Flask, with the following structure:

```
ml_api/
│
├── data/
│   ├── Generated_Recipe_Dataset.csv
│   ├── World_Dishes_Dataset (1).csv
│   └── clustered_dishes.csv
│
├── models/
│   ├── classification_model.py
│   ├── classification_model.pkl
│   ├── classifier_vectorizer.pkl
│   ├── clustering_model.py
│   ├── clustering_model.pkl
│   ├── clustering_vectorizer.pkl
│   ├── kmeans_model.pkl
│   ├── pca.pkl
│   ├── recommendation_model.py
│   ├── recommendation_vectorizer.pkl
│   ├── regression_model.py
│   ├── regression_model.pkl
│   ├── scaler.pkl
│   └── tfidf_vectorizer.pkl
│
├── ml_api.py                # Main Flask API, loads models and exposes endpoints
├── train_clustering.py      # Script for training clustering models
└── train_model.py           # Script for training classification/recommendation/regression models
```

### Core Components

- **ml_api.py**: Entry point for the API server. Loads all ML models and provides endpoints for:
  - `/api/classification`: Predicts dish category from ingredients.
  - `/api/recommendation`: Recommends dishes based on input ingredients.
  - `/api/clustering`: Returns cluster/category info for a dish.
  - `/api/regression`: Predicts calories and finds matching dishes.
- **models/**: Contains Python implementations and serialized (pickle) files for:
  - Classification, clustering, recommendation, and regression models
  - Associated vectorizers, scalers, and PCA objects
- **data/**: CSV datasets used for training and inference.
- **train_clustering.py**, **train_model.py**: Scripts for (re)training respective models.

> Each model class is initialized with a dataset and exposes a `predict` (and/or `recommend`) method for the API.

[Browse ml_api directory →](https://github.com/The-Menufy/IA/tree/main/ml_api)

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/)
- Optional: [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda](https://docs.conda.io/)

### Clone the Repository

```bash
git clone https://github.com/The-Menufy/IA.git
cd IA
```

### Set Up Your Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚦 Usage

### Basic Usage

```bash
cd ml_api
python ml_api.py
```

The API will start (by default) at `http://0.0.0.0:5001/` using Waitress.

### Endpoints

- `POST /api/classification`
  - Input: `{ "ingredients": "string of ingredients" }`
  - Output: `{ "category": "..." }`
- `POST /api/recommendation`
  - Input: `{ "ingredients": ["ingredient1", "ingredient2", ...] }`
  - Output: `[ { "Dish Name": "...", ... }, ... ]`
- `POST /api/clustering`
  - Input: `{ "dish_name": "..." }`
  - Output: `{ "dish_name": "...", "category": "...", "cluster": ... }`
- `POST /api/regression`
  - Input: `{ "features": [Num_Ingredients, Preparation_Time_Minutes, Cooking_Time_Minutes], "avg_cal_per_ingredient": 50 }`
  - Output: `{ "predicted_calories": ..., "matching_dishes": [...] }`

### Command-Line Help

For training scripts, see:
```bash
python train_model.py --help
python train_clustering.py --help
```

---

## ⚙️ Configuration

- Datasets are located in `ml_api/data/`.
- Models and vectorizers are loaded from `ml_api/models/`.
- To use new datasets or retrain models, run the training scripts.

Example configuration variables (see top of `ml_api.py`):

- `BASE_DIR`: Absolute path to `ml_api/`
- `DATA_DIR`: Path to data directory
- Model paths are defined relative to these

---

## 💡 Examples

- **Classify a dish by ingredients:**
    ```bash
    curl -X POST http://localhost:5001/api/classification \
      -H "Content-Type: application/json" \
      -d '{"ingredients": "chicken, rice, spices"}'
    ```
- **Get dish recommendations:**
    ```bash
    curl -X POST http://localhost:5001/api/recommendation \
      -H "Content-Type: application/json" \
      -d '{"ingredients": ["rice", "tomato", "pepper"]}'
    ```
- **Cluster a dish:**
    ```bash
    curl -X POST http://localhost:5001/api/clustering \
      -H "Content-Type: application/json" \
      -d '{"dish_name": "Paella"}'
    ```
- **Calorie regression:**
    ```bash
    curl -X POST http://localhost:5001/api/regression \
      -H "Content-Type: application/json" \
      -d '{"features": [5, 15, 30], "avg_cal_per_ingredient": 60}'
    ```

---

## 📖 API Reference

- Main API code: [`ml_api/ml_api.py`](https://github.com/The-Menufy/IA/blob/main/ml_api/ml_api.py)
- Model implementations: [`ml_api/models/`](https://github.com/The-Menufy/IA/tree/main/ml_api/models)
- Datasets: [`ml_api/data/`](https://github.com/The-Menufy/IA/tree/main/ml_api/data)
- Training scripts: [`ml_api/train_model.py`](https://github.com/The-Menufy/IA/blob/main/ml_api/train_model.py), [`ml_api/train_clustering.py`](https://github.com/The-Menufy/IA/blob/main/ml_api/train_clustering.py)

---

## 🧪 Testing

Run all tests using:

```bash
pytest tests/
```

For code style and linting, use:

```bash
flake8 ml_api/
```

---

## 🤝 Contributing

We welcome contributions! Please review our [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request.

---

## 🗺️ Roadmap

- [ ] API documentation
- [ ] More advanced models and data sources
- [ ] Docker support
- [ ] CI/CD integration
- [ ] Community-contributed examples

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

- **Project Lead:** [@The-Menufy](https://github.com/The-Menufy)
- **Issues & Support:** [GitHub Issues](https://github.com/The-Menufy/IA/issues)
- **Discussions:** [GitHub Discussions](https://github.com/The-Menufy/IA/discussions)

---

## 🙏 Acknowledgements

- Python community
- Open-source ML and data science projects
- Flask, scikit-learn, pandas, and other libraries
- [Add any datasets or contributors here]

---

*Crafted with ❤️ by The-Menufy Team*
