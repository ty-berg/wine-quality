import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("Cleaned data.csv")
print("Initial dataset shape:", df.shape)

# Get all numeric columns in the dataset
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude the target column ('quality') from outlier removal (if desired)
if 'quality' in numeric_cols:
    numeric_features = [col for col in numeric_cols if col != 'quality']
else:
    numeric_features = numeric_cols

print("Numeric features used for outlier detection:", numeric_features)

# Function to compute IQR bounds for a given Series
def compute_iqr_bounds(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return lower_bound, upper_bound

# Remove outliers for each numeric feature (except the target)
df_clean = df.copy()
for col in numeric_features:
    lower_bound, upper_bound = compute_iqr_bounds(df_clean[col])
    initial_shape = df_clean.shape
    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    print(f"After removing outliers in '{col}': from {initial_shape} to {df_clean.shape}")

print("Final dataset shape after outlier removal:", df_clean.shape)

# Save the cleaned dataset to a new CSV file
output_filename = "Cleaned_data_no_outliers.csv"
df_clean.to_csv(output_filename, index=False)
print(f"Cleaned dataset saved as '{output_filename}'")

