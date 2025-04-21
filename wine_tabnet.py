import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# === Load Dataset ===
df = pd.read_csv("Cleaned_data_no_outliers.csv")
X = df.drop(columns=["quality"])
y = df["quality"].values

# === Standardize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Validation/Test Split ===
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === Box–Cox Transform the Target ===
pt = PowerTransformer(method="box-cox")
y_train_boxcox = pt.fit_transform(y_train.reshape(-1, 1))
eval_y = pt.transform(y_test.reshape(-1, 1))

# === Convert Features to float32 (required for TabNet) ===
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# === Initialize TabNet Regressor ===
tabnet_model = TabNetRegressor(
    n_d=16, n_a=16,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',  # or 'sparsemax'
    verbose=0,
    seed=42
)

# === Train the Model ===
tabnet_model.fit(
    X_train=X_train, y_train=y_train_boxcox,
    eval_set=[(X_test, eval_y)],
    eval_metric=["rmse"],
    max_epochs=300,
    patience=30,
    batch_size=256,
    virtual_batch_size=128
)

# === Predict on Test Set ===
preds_boxcox = tabnet_model.predict(X_test)
preds = pt.inverse_transform(preds_boxcox.reshape(-1, 1)).flatten()

# === Evaluate ===
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
rounded_preds = np.round(preds)
accuracy = accuracy_score(y_test, rounded_preds) * 100

print(f"TabNet Test RMSE: {rmse:.4f}")
print(f"TabNet Test R² Score: {r2:.4f}")
print(f"TabNet Test Rounded Accuracy: {accuracy:.2f}%")
