# Wine Quality Prediction Project

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

## Overview

This project focuses on predicting the quality of wine using various machine learning models. It involves data cleaning, outlier removal, and the training of multiple classification models to predict wine quality based on physicochemical features.

## Repository Structure

wine-quality/
├── .DS_Store
├── Cleaned_data_no_outliers.csv
├── remove_outliers.py
├── wine_dummy.py
├── wine_mlp.py
├── wine_mlpxg.py
├── wine_randomforest.py
├── wine_stacked.py
├── wine_xgboost.py
├── curve_random.py
├── README.md
└── requirements.txt

* **.DS\_Store**: (Likely a macOS metadata file, might be good to add to `.gitignore`)
* **`Cleaned data.csv`**: An intermediate cleaned version of the wine quality data obtained from https://github.com/You-sha/Wine-Quality-Prediction/tree/main.
* **`Cleaned_data_no_outliers.csv`**: The final, preprocessed dataset used for training the machine learning models, with outliers removed. This is the key data file for model building.
* **`remove_outliers.py`**: A Python script responsible for identifying and removing outliers from the wine quality data, used to generate `Cleaned_data_no_outliers.csv`.
* **`wine_dummy.py`**: Likely implements a baseline or dummy classification model for comparison.
* **`wine_mlp.py`**: Implements a Multi-Layer Perceptron (MLP) model.
* **`wine_mlpxg.py`**: Combines an MLP with an XGBoost model.
* **`wine_randomforest.py`**: Contains the implementation and training of a Random Forest model.
* **`wine_stacked.py`**: Implements a stacked ensemble model.
* **`wine_xgboost.py`**: Implements an XGBoost model.
* **`curve_random.py`**: Generates curve for Random Forest comparing number of estimators to RMSE.
* **`README.md`**: This file.
* **`requirements.txt`**: Lists the Python packages required.

## Data Source and Preprocessing

The project utilizes wine quality data. The script `remove_outliers.py` plays a crucial role by generating `Cleaned_data_no_outliers.csv`, used by all models.

## Model Implementations

Separate scripts are provided for different models:

* `wine_dummy.py`: Baseline model.
* `wine_mlp.py`: Neural network model.
* `wine_mlpxg.py`: MLP and XGBoost combination.
* `wine_randomforest.py`: Random Forest model.
* `wine_stacked.py`: Stacked ensemble.
* `wine_xgboost.py`: XGBoost model.

Each model script reads from `Cleaned_data_no_outliers.csv`.

## Getting Started

```bash
git clone https://github.com/ty-berg/wine-quality.git
cd wine-quality

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
python remove_outliers.py
```

Run individual models:
```bash
python wine_dummy.py
python wine_mlpxg.py
# etc.
```

## Evaluation Metrics

* **Accuracy**: Correct predictions / Total predictions
* **RMSE**: Square root of the average squared difference between predicted values and actual values
* **R^2**: Proportion of variance in the dependent variable that is predictable from the independent variables.
* **Confusion Matrix**

## Enhancements

* Centralized evaluation script
* Hyperparameter tuning
* Code documentation
* Experiment tracking (e.g., MLflow)
* Jupyter Notebooks for exploration

## Contributing

Please fork the repo, create a new branch, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

* [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* Developers of pandas, scikit-learn, XGBoost