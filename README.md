# Predicting Employee Retention: A Logistic Regression Analysis

A data-driven Logistic Regression model built on 74K+ employee records to proactively predict retention likelihood and strengthen workforce stability.

## Business Objective

The primary goal of this project was to help a mid-sized technology company improve its understanding of employee retention. By developing a predictive model, the project aims to proactively identify employees likely to stay and pinpoint the factors contributing to their loyalty, providing the HR department with actionable insights to strengthen retention strategies and increase overall workforce stability[cite: 1, 2].

## Key Accomplishments & Technical Approach

| Category | Accomplishment |
| :--- | :--- |
| **Model Development** | **Developed a Logistic Regression Model** to predict employee retention, achieving a robust predictive performance through feature engineering and cutoff optimization. |
| **Data Processing** | **Engineered a comprehensive dataset** of 74K+ employee records, including demographics, job satisfaction, and performance metrics, expertly handling **missing values** (imputation) and **categorical encoding**. |
| **Feature Optimization** | Optimized model performance using **Recursive Feature Elimination (RFE)** for feature selection, focusing on the top 15 most influential features to ensure model simplicity and interpretability. |
| **Model Evaluation** | Determined an **optimal probability cutoff** by analyzing ROC, Sensitivity/Specificity, and Precision/Recall trade-offs, fine-tuning the model for maximum business impact. |

## Tech Stack

* **Programming Language:** Python
* **Core Libraries:** Pandas, NumPy
* **Modeling & Evaluation:** Scikit-learn (`LogisticRegression`, `train_test_split`, `metrics`, `precision_recall_curve`, `RFE`, `StandardScaler`), Statsmodels (`Logit`, `VIF`)
* **Data Visualization:** Matplotlib, Seaborn

## Methodology: Step-by-Step Breakdown

### 1. Data Understanding & Cleaning
* **Load and Inspect Data:** Loaded the dataset containing 74,610 rows and 24 columns, examining initial structure and data types.
* **Handle Missing Values:** Addressed missing data in two key numerical columns: 'Distance from Home' and 'Company Tenure (In Months)' by imputing them with the mean value, as the percentage of missing values was low (2.56% and 3.23% respectively).
* **Identify Redundancy:** Inspected categorical features for redundant values and unique identifiers.
* **Drop Columns:** Dropped non-essential or potentially redundant columns (e.g., `Employee ID`, `Company Tenure (In Months)`) to simplify the model.

### 2. Exploratory Data Analysis (EDA) on Training Data
* **Univariate Analysis:** Visualized the distribution of numerical columns (e.g., Age, Monthly Income) using histograms to identify skewness and outliers.
* **Correlation Analysis:** Generated a heatmap to analyze the correlation matrix of numerical features, confirming the relationship between 'Age' and 'Years at Company'.
* **Class Balance:** Checked the distribution of the target variable (`Attrition`), noting that the dataset is imbalanced (Stayed vs. Left).
* **Bivariate Analysis:** Plotted countplots for all categorical columns against the target variable to visualize how factors like 'Job Role', 'Work-Life Balance', and 'Performance Rating' influence retention.

### 3. Feature Engineering
* **Dummy Variable Creation:** Applied `pd.get_dummies()` to convert all remaining categorical features (nominal and ordinal) into numerical representations for both the training and validation sets.
* **Feature Scaling:** Used `StandardScaler` to scale all numerical features in both the training and validation sets, ensuring coefficients would not be dominated by features with larger magnitudes.

### 4. Model Building & Optimization
* **Feature Selection (RFE):** Employed **Recursive Feature Elimination (RFE)** to select the most optimal set of 15 features, ensuring model stability and interpretability.
* **Model Fitting:** Built and fitted the Logistic Regression model using `statsmodels` to evaluate the statistical significance of the chosen predictors (p-values) and check for multicollinearity using the **Variance Inflation Factor (VIF)**.
* **Optimal Cutoff:** Analyzed the **ROC curve** and the trade-offs between accuracy, sensitivity, and specificity at various probability thresholds to determine the most effective cutoff point for predicting retention.

### 5. Prediction and Model Evaluation
* **Validation Set Prediction:** Used the finalized model and optimal cutoff to make predictions on the unseen validation dataset.
* **Performance Metrics:** Evaluated the model's performance on the validation set using a confusion matrix and calculated comprehensive metrics, including: **Accuracy, Sensitivity, Specificity, Precision, and Recall**.
