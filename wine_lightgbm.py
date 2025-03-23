import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("WineQT.csv")
print(df.head())

# Verify that the target column 'quality' exists
if 'quality' in df.columns:
    X = df.drop(columns=['quality','Id'])
    y = df['quality']
else:
    print("Error: 'quality' column not found in the dataset. Please adjust the column names accordingly.")
    exit(1)

# Convert all features to z-scores (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a grid of hyperparameters for LightGBM
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the LGBMRegressor
lgb_reg = lgb.LGBMRegressor(objective='regression', random_state=42, verbose=-1)

# Use GridSearchCV to search for the best hyperparameters (3-fold CV using R² as the scoring metric)
grid_search = GridSearchCV(estimator=lgb_reg,
                           param_grid=param_grid,
                           cv=3,
                           scoring='r2',
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)
print("Best CV R²:", grid_search.best_score_)

# Use the best estimator from grid search for further evaluation
best_lgb = grid_search.best_estimator_

# Evaluate on the validation set
y_val_pred = best_lgb.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)
y_val_pred_round = np.round(y_val_pred)
val_accuracy = np.mean(y_val_pred_round == y_val) * 100

print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R²: {val_r2:.4f}")
print(f"Validation Accuracy (rounded): {val_accuracy:.2f}%")

# Evaluate on the test set
y_test_pred = best_lgb.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
y_test_pred_round = np.round(y_test_pred)
test_accuracy = np.mean(y_test_pred_round == y_test) * 100

print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test Accuracy (rounded): {test_accuracy:.2f}%")

# Plot Actual vs. Predicted wine quality for the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.xlabel("Actual Wine Quality")
plt.ylabel("Predicted Wine Quality")
plt.title("Tuned LightGBM: Actual vs Predicted Wine Quality")
plt.grid(True)
plt.tight_layout()
plt.show()

