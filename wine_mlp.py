import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# === Load dataset ===
df = pd.read_csv("Cleaned_data_no_outliers.csv")
X = df.drop(columns=['quality'])
y = df['quality']

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/val/test split ===
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === Box-Cox transform target ===
pt = PowerTransformer(method='box-cox')
y_train_boxcox = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# === Optuna XGBoost tuning ===
def objective(trial):
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
    model = XGBRegressor(**param)
    scores = cross_val_score(model, X_train, y_train_boxcox, cv=5, scoring='r2', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# === Train best XGB model ===
best_params = study.best_params
best_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'verbosity': 0})
best_xgb = XGBRegressor(**best_params)
best_xgb.fit(X_train, y_train_boxcox)

# === Train MLP model ===
mlp_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])
mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mlp_model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=32, callbacks=[early_stop], verbose=0)

# === Make predictions ===
mlp_preds = mlp_model.predict(X_test).flatten()
xgb_preds_boxcox = best_xgb.predict(X_test)
xgb_preds = pt.inverse_transform(xgb_preds_boxcox.reshape(-1, 1)).flatten()

# === Ensemble: average ===
ensemble_preds = (mlp_preds + xgb_preds) / 2

# === Evaluate ===
rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
r2 = r2_score(y_test, ensemble_preds)
rounded_preds = np.round(ensemble_preds)
accuracy = accuracy_score(y_test, rounded_preds) * 100

print(f"Ensemble Test RMSE: {rmse:.4f}")
print(f"Ensemble Test R²: {r2:.4f}")
print(f"Ensemble Accuracy (rounded): {accuracy:.2f}%")

# === Confusion matrix ===
cm = confusion_matrix(y_test, rounded_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Quality")
plt.ylabel("Actual Quality")
plt.title(f"Ensemble Confusion Matrix (Acc: {accuracy:.2f}%)")
plt.tight_layout()
plt.show()
