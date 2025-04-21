import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Load the dataset
df = pd.read_csv("Cleaned_data_no_outliers.csv")
print("Dataset preview:")
print(df.head())

# Check that the target column 'quality' exists and drop 'Id' if present
if 'quality' in df.columns:
    X = df.drop(columns=['quality', 'Id'], errors='ignore')
    y = df['quality']
else:
    raise ValueError("Error: 'quality' column not found. Adjust column names accordingly.")

# Standardize features (convert to z-scores)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#############################################
# Apply Box–Cox Transformation to the Target
#############################################
# Box–Cox requires strictly positive values.
pt = PowerTransformer(method='box-cox')
y_train_boxcox = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_boxcox = pt.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_boxcox = pt.transform(y_test.values.reshape(-1, 1)).flatten()

#############################################
# Define the Objective Function for Bayesian Optimization
#############################################
def objective(trial):
    # Hyperparameters for XGB base model
    n_estimators_xgb = trial.suggest_int("n_estimators_xgb", 500, 700)
    max_depth_xgb = trial.suggest_int("max_depth_xgb", 10, 30)
    learning_rate_xgb = trial.suggest_float("learning_rate_xgb", 0.01, 0.1, log=True)
    subsample_xgb = trial.suggest_categorical("subsample_xgb", [0.7, 0.8, 1.0])
    colsample_bytree_xgb = trial.suggest_categorical("colsample_bytree_xgb", [0.7, 0.8, 1.0])
    reg_alpha_xgb = trial.suggest_float("reg_alpha_xgb", 0, 1.0)
    reg_lambda_xgb = trial.suggest_float("reg_lambda_xgb", 0, 1.0)
    
    # Hyperparameters for RandomForest base model
    n_estimators_rf = trial.suggest_int("n_estimators_rf", 1000, 1500)
    max_depth_rf = trial.suggest_int("max_depth_rf", 30, 70)
    
    # Hyperparameter for Ridge base model
    alpha_ridge = trial.suggest_float("alpha_ridge", 0.1, 10.0, log=True)
    
    # Hyperparameters for meta model (XGBRegressor)
    n_estimators_meta = trial.suggest_int("n_estimators_meta", 100, 500)
    max_depth_meta = trial.suggest_int("max_depth_meta", 5, 30)
    learning_rate_meta = trial.suggest_float("learning_rate_meta", 0.01, 0.1, log=True)
    
    # Instantiate base models with the trial's parameters:
    base_estimators = [
        ('xgb', XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_estimators_xgb,
            max_depth=max_depth_xgb,
            learning_rate=learning_rate_xgb,
            subsample=subsample_xgb,
            colsample_bytree=colsample_bytree_xgb,
            reg_alpha=reg_alpha_xgb,
            reg_lambda=reg_lambda_xgb,
            random_state=42,
            verbosity=0
        )),
        ('rf', RandomForestRegressor(
            n_estimators=n_estimators_rf,
            max_depth=max_depth_rf,
            random_state=42
        )),
        ('ridge', Ridge(alpha=alpha_ridge))
    ]
    
    # Instantiate meta model:
    meta_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=n_estimators_meta,
        max_depth=max_depth_meta,
        learning_rate=learning_rate_meta,
        random_state=42,
        verbosity=0
    )
    
    # Create the stacking ensemble
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    # Use 5-fold CV on the training set (using the Box–Cox transformed target)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stacking_model, X_train, y_train_boxcox, cv=cv, scoring='r2', n_jobs=-1)
    return scores.mean()

# Create an Optuna study to maximize the CV R² on the transformed target and optimize over 50 trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters from stacking ensemble:", study.best_params)
print("Best CV R² from stacking ensemble:", study.best_value)

#############################################
# Build Final Stacking Ensemble with Best Hyperparameters
#############################################
params = study.best_params

base_estimators_best = [
    ('xgb', XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params["n_estimators_xgb"],
        max_depth=params["max_depth_xgb"],
        learning_rate=params["learning_rate_xgb"],
        subsample=params["subsample_xgb"],
        colsample_bytree=params["colsample_bytree_xgb"],
        reg_alpha=params["reg_alpha_xgb"],
        reg_lambda=params["reg_lambda_xgb"],
        random_state=42,
        verbosity=0
    )),
    ('rf', RandomForestRegressor(
        n_estimators=params["n_estimators_rf"],
        max_depth=params["max_depth_rf"],
        random_state=42
    )),
    ('ridge', Ridge(alpha=params["alpha_ridge"]))
]

meta_model_best = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=params["n_estimators_meta"],
    max_depth=params["max_depth_meta"],
    learning_rate=params["learning_rate_meta"],
    random_state=42,
    verbosity=0
)

stacking_model_best = StackingRegressor(
    estimators=base_estimators_best,
    final_estimator=meta_model_best,
    cv=5,
    n_jobs=-1
)

# Train the final stacking ensemble on the training set (Box–Cox transformed target)
stacking_model_best.fit(X_train, y_train_boxcox)

#############################################
# Evaluation: Inverse Transformation on Predictions
#############################################
def evaluate_model(model, X, y_true):
    # Predict in transformed space then invert the Box–Cox transformation
    y_pred_boxcox = model.predict(X)
    y_pred = pt.inverse_transform(y_pred_boxcox.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    acc = accuracy_score(y_true, np.round(y_pred)) * 100
    return y_pred, rmse, r2, acc

# Evaluate on validation set
y_val_pred, val_rmse, val_r2, val_accuracy = evaluate_model(stacking_model_best, X_val, y_val)
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R²: {val_r2:.4f}")
print(f"Validation Accuracy (rounded): {val_accuracy:.2f}%")

# Evaluate on test set
y_test_pred, test_rmse, test_r2, test_accuracy = evaluate_model(stacking_model_best, X_test, y_test)
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test Accuracy (rounded): {test_accuracy:.2f}%")

#############################################
# Generate Confusion Matrix on All Data
#############################################
y_all_pred_boxcox = stacking_model_best.predict(X_scaled)
y_all_pred = pt.inverse_transform(y_all_pred_boxcox.reshape(-1, 1)).flatten()
y_all_pred_round = np.round(y_all_pred)
overall_accuracy = accuracy_score(y, y_all_pred_round) * 100
print("Overall Accuracy on all data:", overall_accuracy)

cm = confusion_matrix(y, y_all_pred_round)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Quality")
plt.ylabel("Actual Quality")
plt.title(f"Confusion Matrix (Stacking Ensemble) (Acc: {overall_accuracy:.2f}%)")
plt.tight_layout()
plt.show()
