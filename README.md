# Predicting Ultimate Claim Amounts from FNOL Data 

This project was developed as a Junior Data Scientist case study for Hastings Direct. The objective was to build a predictive model that estimates the ultimate claim cost of head-on collision motor insurance claims using only First Notification Of Loss (FNOL) data.

---

## 📌 Problem Statement

Insurance companies often learn the final cost of a claim years after the incident. This delay impacts pricing and reserving decisions. By predicting the ultimate cost at the FNOL stage, insurers can price risk better and detect potential fraud early.

---

## 📂 Dataset

The dataset includes FNOL records of head-on collisions and final incurred values. Features include:
- Policyholder demographics
- Time and day of the incident
- Location and vehicle info
- Third-party involvement (TP_*)
- Binary flags (e.g., WitnessPresent, PoliceReportFiled)

---

## 🔄 Data Preparation and Feature Engineering

This project involved comprehensive data preprocessing, transformation, and explainability techniques to ensure reliable claim cost prediction.

### 🧹 1. Data Cleaning and Preprocessing

- **Duplicate Removal:**  
  Removed exact duplicate rows using `drop_duplicates()`.

- **Missing Value Handling:**  
  - Categorical fields were filled with `'Unknown'`.  
  - Numerical columns were imputed using the mean.

- **Irrelevant Columns Dropped:**  
  Removed `ClaimNumber`, `Date`, and `AccidentDate`.

- **Time Extraction:**  
  Created a new `Hour` column from the `Time` field to study risk periods.

- **Ordinal Mapping of Days:**  
  Converted `DayOfWeek` into integers (Monday = 0, ..., Sunday = 6).

- **Label Encoding:**  
  Transformed categorical variables into numeric values.

### 🧱 2. Feature Engineering

- **Weekend Indicator (`Weekend`):**  
  1 if the accident occurred on Saturday or Sunday, else 0.

- **Third-Party Aggregation (`TotalTP`):**  
  Summed all `TP_*` columns to capture total third-party involvement.

- **Dropped Original `TP_*` Columns:**  
  Reduced dimensionality and noise after aggregation.

---

### 🔁 3. Power Transformation

- **Why:**  
  The target variable `Incurred` was right-skewed.

- **How:**  
  Used `PowerTransformer` (Yeo-Johnson method) from `sklearn.preprocessing`.

- **Impact:**  
  Normalized the target, improved learning, and reduced sensitivity to outliers.

---

### 📊 4. SHAP Values – Model Explainability

- **Goal:**  
  Interpret the trained model and understand feature contributions.

- **Method:**  
  - Used `shap.TreeExplainer()` on Random Forest model.  
  - Visualized global feature importance using summary and bar plots.

- **Results:**  
  Top features: `TotalTP`, `Location`, `Hour`, and `Weekend`.  
  SHAP confirmed the model’s decisions were consistent with domain insights.

---

## 🧠 Model Development

- Trained multiple models:
  - Linear Regression
  - Lasso/Ridge Regression
  - Decision Tree
  - XG Boost
  - **Random Forest Regressor** (final choice)

- **Model Metrics:**
  - R² Score: ~0.87
  - RMSE: ~1550

---

## 📈 Business Impact

- Enables earlier estimation of future claim payouts.
- Helps in fraud detection and reserving strategy.
- Supports real-time FNOL triage and decision-making.

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn
- SHAP
- Matplotlib, Seaborn
- Jupyter Notebook

---

