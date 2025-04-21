import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Cleaned_data_no_outliers.csv")
print("Dataset preview:")
print(df.head())

# Check that the target column 'quality' exists and drop the 'Id' column if present
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
# Train DummyRegressor (Trivial Baseline)
#############################################
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)

# Evaluate on training set
y_train_pred = dummy.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
print("Training Metrics:")
print(f"RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")

# Evaluate on validation set
y_val_pred = dummy.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)
print("Validation Metrics:")
print(f"RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

# Evaluate on test set
y_test_pred = dummy.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
print("Test Metrics:")
print(f"RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

# Compute "rounded accuracy" on test set by rounding predictions to the nearest integer
y_test_pred_round = np.round(y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred_round) * 100
print(f"Test Rounded Accuracy: {test_accuracy:.2f}%")

#############################################
# Generate Confusion Matrix on All Data
#############################################
# Predict on all data
y_all_pred = dummy.predict(X_scaled)
y_all_pred_round = np.round(y_all_pred)
overall_accuracy = accuracy_score(y, y_all_pred_round) * 100
print("Overall Accuracy on all data:", overall_accuracy)

# Compute the confusion matrix (comparing rounded predictions to true quality values)
cm = confusion_matrix(y, y_all_pred_round)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Quality")
plt.ylabel("Actual Quality")
plt.title(f"Confusion Matrix for Dummy Regressor (Acc: {overall_accuracy:.2f}%)")
plt.tight_layout()
plt.show()
