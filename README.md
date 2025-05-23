🔧 1. Data Cleaning and Preprocessing
Duplicate Removal:
Removed duplicate rows to prevent bias in training.

Missing Value Handling:

Categorical columns: filled with 'Unknown'.

Numerical columns: imputed using column mean.

Irrelevant Columns Dropped:

ClaimNumber, Date, AccidentDate were removed for not adding predictive value.

Time Conversion:

Extracted hour component from Time and created a new Hour column.

Day Mapping:

Converted DayOfWeek to ordinal integers (e.g., Monday = 0).

Categorical Encoding:

Used Label Encoding for all categorical variables to prepare data for tree-based models.

🧱 2. Feature Engineering
Weekend Flag:
Created a binary column Weekend for Saturday and Sunday crashes.

Working Hours Flag:
Added WorkingHours column indicating whether the crash occurred between 9 AM and 5 PM.

Third-Party Aggregation:
Summed all TP_* columns into a single TotalTP feature and dropped the originals.

🔁 3. Power Transformation
Motivation:
The target variable Incurred had a right-skewed distribution. This non-normality can negatively affect model performance—especially for regression tasks.

Transformation Used:
Applied PowerTransformer from sklearn.preprocessing (Yeo-Johnson method by default) to normalize the distribution of Incurred.

Impact:

Improved the model's ability to learn patterns in the data.

Reduced the effect of extreme values or heavy skew.

📊 4. SHAP Values – Model Explainability
Purpose:
After model training, you used SHAP (SHapley Additive exPlanations) to explain how features contributed to the model’s predictions.

Implementation:

Used shap.TreeExplainer() with the trained Random Forest Regressor.

Plotted summary plot and feature impact bars to interpret global model behavior.

Key Insights from SHAP:

TotalTP, Location, Hour, and Weekend were among the top features influencing the final claim value.

SHAP values helped validate that your engineered features were relevant and impactful.

Provided transparency for business stakeholders and aligned with the goal of making the model actionable in a production setting.

✅ Summary
These transformations and interpretability tools:

Improved model performance (R² ≈ 0.87).

Reduced skew in the target variable using PowerTransformer.

Validated feature importance with SHAP, providing insights critical for deploying the model responsibly.
