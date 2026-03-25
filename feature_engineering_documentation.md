
# 🔧 Feature Engineering Documentation — Customer Churn Prediction

## 1. Purpose
Feature engineering was applied to enrich the dataset with new variables that capture customer behavior, spending patterns, and contract characteristics. These features aim to improve the predictive power of the churn model by highlighting hidden relationships in the raw data.

---

## 2. Engineered Features

### 2.1 Customer Lifetime Value (CLV)
- **Formula:**  
  \[
  \text{CLV} = \text{MonthlyCharges} \times \text{Tenure}
  \]
- **Rationale:**  
  Represents the total revenue a customer has generated during their tenure. Higher CLV often indicates loyal, long-term customers less likely to churn.

---

### 2.2 Payment Efficiency
- **Formula:**  
  \[
  \text{PaymentEfficiency} = \frac{\text{TotalCharges}}{\text{Tenure} + 1}
  \]
- **Rationale:**  
  Measures how consistently customers pay relative to their tenure. Low efficiency may indicate irregular payments or billing issues, which can correlate with churn.

---

### 2.3 Average Monthly Spend
- **Formula:**  
  \[
  \text{AverageMonthlySpend} = \frac{\text{TotalCharges}}{\text{Tenure} + 1}
  \]
- **Rationale:**  
  Captures the average monthly billing amount. Customers with unusually high or low averages may have different churn behaviors compared to typical spenders.

---

### 2.4 Tenure Group
- **Formula:**  
  Bucketed tenure ranges:  
  - 0–12 months → `"0-12"`  
  - 13–24 months → `"13-24"`  
  - 25–48 months → `"25-48"`  
  - 49–72 months → `"49-72"`  
- **Rationale:**  
  Groups customers by loyalty stage. Early-tenure customers are more likely to churn, while long-tenure customers often show stability.

---

### 2.5 IsSenior
- **Formula:**  
  \[
  \text{IsSenior} = 
  \begin{cases} 
  1 & \text{if SeniorCitizen = 1} \\ 
  0 & \text{otherwise} 
  \end{cases}
  \]
- **Rationale:**  
  Converts the `SeniorCitizen` flag into a clearer binary feature. Senior customers may have different service usage patterns and churn risks compared to younger customers.

---

## 3. Additional Notes
- All engineered features were added **before scaling** to ensure proper normalization.  
- Outliers in `MonthlyCharges` and `TotalCharges` were capped using the **IQR method** to prevent distortion in engineered features.  
- Features were validated through **correlation analysis** and **RandomForest importance ranking** to confirm their relevance.  

---

