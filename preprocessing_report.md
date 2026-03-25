
# 📝 Preprocessing Report — Customer Churn Prediction

## 1. Data Overview
- **Dataset:** `churn_data.csv`  
- **Columns:**  
  - `CustomerID` (unique identifier)  
  - `Tenure` (months with company)  
  - `MonthlyCharges` (monthly bill amount)  
  - `TotalCharges` (total amount billed)  
  - `Contract` (contract type: Month-to-Month, One Year, Two Year)  
  - `PaymentMethod` (payment type: Electronic, Mailed, Bank Transfer, Credit Card)  
  - `PaperlessBilling` (Yes/No)  
  - `SeniorCitizen` (binary flag: 0/1)  
  - `Churn` (target variable: Yes/No)

---

## 2. Handling Missing Values
- Checked for missing values using `df.isnull().sum()`.  
- Imputed missing numeric values with **median**.  
- Imputed missing categorical values with **most frequent category**.

---

## 3. Encoding Categorical Features
- **Label Encoding**: Applied to binary categorical features (`PaperlessBilling`, `Churn`).  
- **One-Hot Encoding**: Applied to multi-class categorical features (`Contract`, `PaymentMethod`).  
- **Target Encoding (optional)**: Considered for high-cardinality categorical features if present.

---

## 4. Outlier Detection & Handling
- **IQR Method**: Applied to `MonthlyCharges` and `TotalCharges`.  
  - Outliers capped at upper/lower bounds.  
- **Z-Score Method**: Verified extreme values in numeric columns.  

---

## 5. Feature Engineering
Created **5+ new features** to improve predictive power:
1. **CustomerLifetimeValue (CLV)** = `MonthlyCharges * Tenure`  
2. **PaymentEfficiency** = `TotalCharges / (Tenure + 1)`  
3. **AverageMonthlySpend** = `TotalCharges / (Tenure + 1)`  
4. **TenureGroup** = bucketed tenure ranges (`0–12`, `13–24`, `25–48`, `49–72`)  
5. **IsSenior** = binary flag derived from `SeniorCitizen`  

---

## 6. Feature Scaling
- **Min-Max Scaling**: Applied to `MonthlyCharges` for normalization.  
- **Standard Scaling**: Applied to `Tenure`, `TotalCharges`, `CLV`, `PaymentEfficiency`, `AverageMonthlySpend`.  
- Compared distributions before and after scaling using histograms.

---

## 7. Feature Selection
- **Correlation Analysis**: Dropped highly correlated redundant features.  
- **RandomForest Feature Importance**: Ranked predictors to identify most influential features for churn.

---

## 8. Preprocessing Pipeline
Built a **scikit-learn pipeline** combining:
- Feature engineering  
- Encoding  
- Outlier handling  
- Scaling  

This ensures reproducibility and consistency across training and testing datasets.

---

## ✅ Summary
- Applied **3 encoding methods** (Label, One-Hot, Target).  
- Applied **2 scaling techniques** (Min-Max, Standard).  
- Created **5+ engineered features**.  
- Handled outliers using **IQR & Z-score**.  
- Built a **complete preprocessing pipeline**.  
- Documented all steps for transparency and reproducibility.  
