import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
import optuna

# Load dataset
df = pd.read_csv("Cleaned_data_no_outliers.csv")
X = df.drop(columns=['quality'])
y = df['quality']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Box-Cox target transform
pt = PowerTransformer(method='box-cox')
y_train_boxcox = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# Optuna search
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

# Final XGB model
best_params = study.best_params
best_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'verbosity': 0})
xgb_model = XGBRegressor(**best_params)
xgb_model.fit(X_train, y_train_boxcox)

# Augment features with XGB predictions
xgb_train_preds = pt.inverse_transform(xgb_model.predict(X_train).reshape(-1, 1))
xgb_test_preds = pt.inverse_transform(xgb_model.predict(X_test).reshape(-1, 1))
X_train_aug = np.hstack([X_train, xgb_train_preds])
X_test_aug = np.hstack([X_test, xgb_test_preds])

# MLP model on augmented input
mlp_model = Sequential([
    Input(shape=(X_train_aug.shape[1],)),
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])

mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mlp_model.fit(X_train_aug, y_train, validation_split=0.2, epochs=200, batch_size=32, callbacks=[early_stop], verbose=0)

# Evaluate
aug_preds = mlp_model.predict(X_test_aug).flatten()
rmse = np.sqrt(mean_squared_error(y_test, aug_preds))
r2 = r2_score(y_test, aug_preds)
accuracy = accuracy_score(y_test, np.round(aug_preds)) * 100

print(f"Hybrid Model RMSE: {rmse:.4f}")
print(f"Hybrid Model R²: {r2:.4f}")
print(f"Hybrid Model Accuracy (rounded): {accuracy:.2f}%")
