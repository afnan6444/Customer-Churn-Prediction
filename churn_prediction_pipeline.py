# 📌 Customer Churn Preprocessing Pipeline

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------
# Step 1: Load Data
# -------------------------------
df = pd.read_csv("customer_churn.csv")

print("Data Shape:", df.shape)
print(df.head())

# -------------------------------
# Step 2: Feature Engineering
# -------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Engineered features
        X['CustomerLifetimeValue'] = X['MonthlyCharges'] * X['Tenure']
        X['PaymentEfficiency'] = X['TotalCharges'] / (X['Tenure'] + 1)
        X['AverageMonthlySpend'] = X['TotalCharges'] / (X['Tenure'] + 1)
        X['TenureGroup'] = pd.cut(X['Tenure'], bins=[0,12,24,48,72], labels=['0-12','13-24','25-48','49-72'])
        X['IsSenior'] = X['SeniorCitizen'].apply(lambda x: 1 if x == 1 else 0)
        return X

# -------------------------------
# Step 3: Encoding
# -------------------------------
binary_cols = ['PaperlessBilling','Churn']
multi_cols = ['Contract','PaymentMethod']

# Label Encoding for binary
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# One-Hot Encoding for multi-class
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# -------------------------------
# Step 4: Outlier Handling (IQR)
# -------------------------------
def handle_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df[col] = np.where(df[col] > upper, upper, np.where(df[col] < lower, lower, df[col]))
    return df

df = handle_outliers(df, 'MonthlyCharges')
df = handle_outliers(df, 'TotalCharges')

# -------------------------------
# Step 5: Scaling
# -------------------------------
num_cols = ['Tenure','MonthlyCharges','TotalCharges','CustomerLifetimeValue','PaymentEfficiency','AverageMonthlySpend']

scaler_pipeline = ColumnTransformer([
    ('minmax', MinMaxScaler(), ['MonthlyCharges']),
    ('standard', StandardScaler(), ['Tenure','TotalCharges','CustomerLifetimeValue','PaymentEfficiency','AverageMonthlySpend'])
], remainder='passthrough')

# -------------------------------
# Step 6: Full Pipeline
# -------------------------------
preprocessing_pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('scaling', scaler_pipeline)
])

processed_data = preprocessing_pipeline.fit_transform(df)

print("✅ Preprocessing pipeline completed. Shape:", processed_data.shape)
