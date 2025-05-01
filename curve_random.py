import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_data_no_outliers.csv")

# Drop unused columns and define features/target
X = df.drop(columns=['quality'])
y = df['quality']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

# List of estimators to evaluate
estimator_values = [50, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000]
rmse_results = []

for n in estimator_values:
    rf = RandomForestRegressor(
        n_estimators=n,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_results.append(rmse)
    print(f"n_estimators = {n}, RMSE = {rmse:.4f}")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(estimator_values, rmse_results, marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("RMSE")
plt.title("Random Forest: RMSE vs. Number of Estimators")
plt.grid(True)
plt.tight_layout()
plt.show()
