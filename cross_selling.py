import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer

# Load the datasets
df = pd.read_csv('C:\\Users\\anuti\\OneDrive\\Desktop\\internship\\Cross Selling\\Test.csv')  

# Select numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

### Initial Summary ###

# Check for missing values
missing_values_before = df.isna().sum()

# Check for skewness
skewness_before = df[numeric_cols].skew()

# Detect outliers using the IQR method
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    return outliers

outliers_iqr_before = df[numeric_cols].apply(detect_outliers_iqr, axis=0)
outliers_count_iqr_before = outliers_iqr_before.sum()

# Detect outliers using Z-score
z_scores_before = np.abs(stats.zscore(df[numeric_cols].dropna()))
outliers_zscore_before = (z_scores_before > 3).sum(axis=0)

### Handling Missing Values ###

# Technique 1: Imputation with Mean/Median/Mode
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)  # Mean imputation for numeric columns

for col in non_numeric_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)  # Mode imputation for non-numeric columns

# Technique 2: Model-Based Imputation (KNN)
# Select columns for KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

### Handling Outliers ###

# Technique 1: Clipping
def clip_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data.clip(lower_bound, upper_bound)

df[numeric_cols] = df[numeric_cols].apply(clip_outliers)

# Technique 2: Transformation (Log Transformation)
# Apply log transformation to reduce the effect of outliers
df[numeric_cols] = df[numeric_cols].apply(lambda x: np.log1p(x))

### After Treatment: Summary Checks ###

# Check for missing values
missing_values_after = df.isna().sum()

# Check for skewness
skewness_after = df[numeric_cols].skew()

# Detect outliers using the IQR method after treatment
outliers_iqr_after = df[numeric_cols].apply(detect_outliers_iqr, axis=0)
outliers_count_iqr_after = outliers_iqr_after.sum()

# Detect outliers using Z-score after treatment
z_scores_after = np.abs(stats.zscore(df[numeric_cols].dropna()))
outliers_zscore_after = (z_scores_after > 3).sum(axis=0)

### Summary Comparison ###

# Combine the before and after summaries into DataFrames
summary_missing_values = pd.DataFrame({
    'Before Treatment': missing_values_before,
    'After Treatment': missing_values_after
})
summary_skewness = pd.DataFrame({
    'Before Treatment': skewness_before,
    'After Treatment': skewness_after
})
summary_outliers_iqr = pd.DataFrame({
    'Before Treatment': outliers_count_iqr_before,
    'After Treatment': outliers_count_iqr_after
})
summary_outliers_zscore = pd.DataFrame({
    'Before Treatment': outliers_zscore_before,
    'After Treatment': outliers_zscore_after
})

# Print the summaries
print("\nMissing Values Summary:\n", summary_missing_values[summary_missing_values.sum(axis=1) > 0])
print("\nSkewness Summary:\n", summary_skewness[(summary_skewness['Before Treatment'].abs() > 1) | (summary_skewness['After Treatment'].abs() > 1)])
print("\nOutliers (IQR method) Summary:\n", summary_outliers_iqr[summary_outliers_iqr.sum(axis=1) > 0])
print("\nOutliers (Z-score method) Summary:\n", summary_outliers_zscore[summary_outliers_zscore.sum(axis=1) > 0])
#print("\nProcessed DataFrame:\n", df.head())