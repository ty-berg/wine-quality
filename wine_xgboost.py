import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Load the dataset
df = pd.read_csv("Cleaned_data_no_outliers.csv")
print(df.head())

# Check that the target column 'quality' exists and drop the 'Id' column if present
if 'quality' in df.columns:
    X = df.drop(columns=['quality'])
    y = df['quality']
else:
    print("Error: 'quality' column not found. Adjust column names accordingly.")
    exit(1)

# Standardize features (convert to z-scores)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#############################################
# Apply Box–Cox Transformation to the Target
#############################################
# PowerTransformer with method 'box-cox' requires strictly positive values.
pt = PowerTransformer(method='box-cox')
y_train_boxcox = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_boxcox = pt.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_boxcox = pt.transform(y_test.values.reshape(-1, 1)).flatten()

#############################################
# Bayesian Optimization (Optuna) for XGBRegressor
#############################################
def objective(trial):
    # Define the hyperparameter search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 700),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 1.0]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.7, 0.8, 1.0]),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'objective': 'reg:squarederror',
        'random_state': 42,
        'verbosity': 0
    }
    
    # Initialize the XGBRegressor with the trial's parameters and train on the box-cox transformed target
    model = XGBRegressor(**param)
    # Use 3-fold cross-validation on the training set (using transformed target)
    scores = cross_val_score(model, X_train, y_train_boxcox, cv=5, scoring='r2', n_jobs=-1)
    return scores.mean()

# Create an Optuna study to maximize the CV R² on the transformed target and optimize over 100 trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters found:", study.best_params)
print("Best CV R²:", study.best_value)

# Retrieve best parameters and update them with fixed settings
best_params = study.best_params
best_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'verbosity': 0})

# Initialize the final XGBRegressor model with the best parameters
best_xgb = XGBRegressor(**best_params)

# Train the final model on the training set using the box-cox transformed target
best_xgb.fit(X_train, y_train_boxcox)

#############################################
# Evaluation: Inverse Transformation on Predictions
#############################################
# Predict on the validation set and invert the Box–Cox transformation
y_val_pred_boxcox = best_xgb.predict(X_val)
y_val_pred = pt.inverse_transform(y_val_pred_boxcox.reshape(-1, 1)).flatten()
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R²: {val_r2:.4f}")

# Compute "accuracy" on the validation set by rounding predictions to the nearest integer
y_val_pred_round = np.round(y_val_pred)
val_accuracy = np.mean(y_val_pred_round == y_val) * 100
print(f"Validation Accuracy (rounded): {val_accuracy:.2f}%")

# Predict on the test set and invert the transformation
y_test_pred_boxcox = best_xgb.predict(X_test)
y_test_pred = pt.inverse_transform(y_test_pred_boxcox.reshape(-1, 1)).flatten()
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Compute "accuracy" on the test set by rounding predictions
y_test_pred_round = np.round(y_test_pred)
test_accuracy = np.mean(y_test_pred_round == y_test) * 100
print(f"Test Accuracy (rounded): {test_accuracy:.2f}%")

#############################################
# Generate Confusion Matrix on All Data
#############################################
# Predict on all standardized data and invert the transformation
y_all_pred_boxcox = best_xgb.predict(X_scaled)
y_all_pred = pt.inverse_transform(y_all_pred_boxcox.reshape(-1, 1)).flatten()
y_all_pred_round = np.round(y_all_pred)
overall_accuracy = accuracy_score(y, y_all_pred_round) * 100
print("Overall Accuracy on all data:", overall_accuracy)

# Compute the confusion matrix (comparing rounded predictions to true quality values)
cm = confusion_matrix(y, y_all_pred_round)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Quality")
plt.ylabel("Actual Quality")
plt.title(f"Confusion Matrix for Final Model on All Data (Acc: {overall_accuracy:.2f}%)")
plt.tight_layout()
plt.show()
