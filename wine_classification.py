import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import optuna

# === Load and preprocess dataset ===
df = pd.read_csv("Cleaned_data_no_outliers.csv")
X = df.drop(columns=['quality'])
y = df['quality']

# Remap quality scores to start at 0
unique_labels = sorted(y.unique())
label_map = {val: idx for idx, val in enumerate(unique_labels)}
y = y.map(label_map)
num_classes = len(unique_labels)

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/val/test split ===
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === XGBoost Optuna Optimization ===
def xgb_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'random_state': 42,
        'n_jobs': -1
    }

    model = XGBClassifier(**param)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=3)
    return 1.0 - scores.mean()  # minimize 1 - accuracy

xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=50)

best_xgb_params = xgb_study.best_params
best_xgb_params.update({
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'objective': 'multi:softprob',
    'random_state': 42,
    'n_jobs': -1
})
xgb_model = XGBClassifier(**best_xgb_params)
xgb_model.fit(X_train, y_train)

# === Generate augmented features ===
xgb_train_probs = xgb_model.predict_proba(X_train)
xgb_test_probs = xgb_model.predict_proba(X_test)
X_train_aug = np.hstack([X_train, xgb_train_probs])
X_test_aug = np.hstack([X_test, xgb_test_probs])

# === MLP Optuna Optimization ===
def mlp_objective(trial):
    tf.keras.backend.clear_session()

    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_int("units", 64, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = Sequential()
    model.add(Input(shape=(X_train_aug.shape[1],)))
    for _ in range(n_layers):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.get(optimizer_name)
    optimizer.learning_rate = learning_rate

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train_aug, y_train, validation_split=0.2,
              epochs=100, batch_size=batch_size, verbose=0, callbacks=[early_stop])

    preds = model.predict(X_test_aug)
    pred_labels = np.argmax(preds, axis=1)
    return 1 - accuracy_score(y_test, pred_labels)

mlp_study = optuna.create_study(direction='minimize')
mlp_study.optimize(mlp_objective, n_trials=50)

# === Train final MLP model ===
best_mlp_params = mlp_study.best_params
final_model = Sequential()
final_model.add(Input(shape=(X_train_aug.shape[1],)))
for _ in range(best_mlp_params["n_layers"]):
    final_model.add(Dense(best_mlp_params["units"], activation='relu'))
    final_model.add(Dropout(best_mlp_params["dropout_rate"]))
final_model.add(Dense(num_classes, activation='softmax'))

final_optimizer = tf.keras.optimizers.get(best_mlp_params["optimizer"])
final_optimizer.learning_rate = best_mlp_params["learning_rate"]

final_model.compile(optimizer=final_optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

final_model.fit(X_train_aug, y_train, validation_split=0.2,
                epochs=200, batch_size=best_mlp_params["batch_size"],
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                verbose=0)

# === Evaluate final model ===
final_preds = final_model.predict(X_test_aug)
final_pred_labels = np.argmax(final_preds, axis=1)
final_acc = accuracy_score(y_test, final_pred_labels)

print(f"\n Final Classification Accuracy: {final_acc * 100:.2f}%")
