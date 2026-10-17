# Predicting Workforce Volatility: An Executive Framework for AI-Driven Layoff Risk

**University of San Diego — Master of Science in Applied Artificial Intelligence (MS-AAI)**  
**Course:** AAI-500: Probability and Statistics for Artificial Intelligence  
**Final Project — Team 7**

---

## Project Overview

### 1. Project Scope & Research Question

* **Title:** *Predicting Workforce Volatility: An Executive Framework for AI-Driven Layoff Risk*
* **Core Research Question:** *"What is an employee's risk of layoff based on their years of experience, age, education level, and industry?"*
* **Target Variable ($y$):** `Layoff_Risk` modeled as an ordinal multi-class outcome:
  * `0`: **Low Risk**
  * `1`: **Medium Risk**
  * `2`: **High Risk**
* **Context & Framing:** Utilizes a structured 20,000-record dataset framed as an **HR decision-support simulation** to evaluate employee vulnerability and protective career factors during enterprise AI adoption.

---

### 2. Feature Architecture

* **Core Predictors ($X$):**
  * `Age`: Continuous numerical variable ($z$-score standardized).
  * `Years_of_Experience`: Continuous numerical variable ($z$-score standardized).
  * `Education_Level`: Ordinal integer encoding capturing credentialing hierarchy:
    * `High School: 0`, `Bachelor's: 1`, `Master's: 2`, `PhD: 3`
  * `Industry`: Nominal categorical feature one-hot encoded with `drop_first=True`, explicitly designating **`IT`** as the dropped reference baseline to anchor regression odds ratios ($\exp(\beta)$). Indicator features represent contrasts against the tech benchmark:
    * `Industry_Education`, `Industry_Finance`, `Industry_Healthcare`, `Industry_Logistics`, `Industry_Manufacturing`, `Industry_Retail`, `Industry_Telecom`
* **Engineered & Domain Indices:**
  * `Automation_Ratio`: Evaluates task exposure by comparing routine task burden relative to creativity requirements.
  * `Shield_Index`: Evaluates how human interaction levels and professional tenure amplify protection against displacement.

---

### 3. Data Preprocessing & Validation Pipeline

* **Integrity & Sanity Audits:**
  * Deduplication verification confirms zero redundant employee records.
  * Career tenure domain sanity constraint enforced: $(\text{Age} - \text{Years of Experience}) \ge 16$, validating that no career history reflects commencement prior to standard legal working age.
  * Comprehensive null audit confirms zero missing values across all 20,000 observations.
* **Stratified 3-Way Data Partitioning:**
  * **Training Set:** 70% ($N = 16,000$)
  * **Test Set:** 15% ($N = 4,000$)
  * Stratification preserves identical target class distribution across all three splits (~33% Low, ~33% Medium, ~34% High).
* **Leakage-Free Feature Scaling:**
  * `StandardScaler` is fitted **strictly on `X_train`** ($\mu_{\text{train}}, \sigma_{\text{train}}$) and applied downstream to transform validation and test sets, maintaining complete out-of-sample data isolation.
* **Processed Data Artifacts:**
  * All 6 partitioned and standardized modeling sets are exported to the [`Data/`](file:///c:/Users/19496/OneDrive/Desktop/MSAAI/AntiGravity%20Final%20Proejct/AAI500-Final-Project/Data) directory (`processed_X_train.csv`, `processed_X_val.csv`, `processed_X_test.csv`, `processed_y_train.csv`, `processed_y_val.csv`, `processed_y_test.csv`).

---

### 4. Analytical Exploratory Data Analysis (EDA) Highlights

Exploratory analyses conducted strictly on the training partition ($N = 14,000$, zero data leakage) reveal four core empirical takeaways:

1. **Seniority and Career Tenure Buffering:**
   * Employee age is invariant across risk categories (median 40.0–41.0 years), indicating that chronological age alone is not a primary driver of vulnerability.
   * Career tenure demonstrates a clear protective effect: workers classified as **High Layoff Risk** possess a median of **6.0 years of experience**, compared to **7.0 years** for Medium Risk and **8.0 years** for Low Risk employees.
2. **Credentialing Protective Gradient:**
   * Educational attainment exhibits an inverse monotonic relationship with layoff vulnerability. High School educated workers face **45.2% High Risk** (vs. 22.5% Low Risk), whereas PhD holders experience only **20.0% High Risk** (and **49.6% Low Risk**). Advanced degrees more than halve high-risk exposure.
3. **Sector Volatility Disparity Relative to Baseline IT:**
   * Operational and supply-chain sectors exhibit the highest vulnerability: **Manufacturing (48.2% High Risk)**, **Retail (42.3%)**, and **Logistics (40.9%)**.
   * Essential human services demonstrate superior resilience: **Healthcare (20.6% High Risk)** and **Education (20.0% High Risk)**.
   * The regression reference baseline, **IT (27.4% High Risk)**, occupies an optimal intermediate position to capture both risk premiums and discounts.
4. **Collinearity Diagnostic Verification:**
   * Pearson correlation between continuous predictors `Age` and `Years_of_Experience` is $r = 0.384$, well below the conventional diagnostic threshold ($|r| < 0.80$).
   * `Education_Level` exhibits virtually zero correlation with age ($r = 0.002$) and tenure ($r = -0.001$), confirming the absence of harmful multicollinearity for subsequent regression modeling.

---

### 5. Benchmarking Model Suite

A 4-model suite balancing statistical inference, interpretability, and predictive performance:

1. **Multinomial Logistic Regression:** Primary parametric baseline calculating parameter coefficients ($\beta$), $p$-values, confidence intervals, and explicit **Odds Ratios ($e^\beta$)** relative to IT.
2. **Gaussian Naïve Bayes:** Diagnostic probabilistic baseline evaluating conditional independence assumptions on mixed workforce attributes.
3. **Random Forest Classifier:** Non-parametric ensemble bagging model capturing non-linear tenure/education interactions and Gini feature importances.
4. **Gradient Boosting Classifier (XGBoost):** Ensemble boosting model optimizing multi-class log-loss, Macro F1-score, and SHAP explainability.

---

### 6. Repository Architecture

```text
AAI500-Final-Project/
│
├── Data/
│   ├── Raw-AI-Dataset.csv             # Raw 20,000-record dataset
│   ├── processed_X_train.csv          # Scaled training predictors (14,000 x 10)
│   ├── processed_X_val.csv            # Scaled validation predictors (3,000 x 10)
│   ├── processed_X_test.csv           # Scaled test predictors (3,000 x 10)
│   ├── processed_y_train.csv          # Training target labels (14,000 x 1)
│   ├── processed_y_val.csv            # Validation target labels (3,000 x 1)
│   └── processed_y_test.csv           # Test target labels (3,000 x 1)
│
├── notebooks/
│   ├── data_cleaning_merged.ipynb     # Master end-to-end data cleaning, encoding, splitting, and Section 3 EDA
│   ├── Data-Cleaning.ipynb            # Team member data cleaning notebook
│   ├── data_cleaning_Saman_Version.ipynb # Alternative team cleaning reference
│   └── Final_Project_Appendix.ipynb   # Technical report code appendix
│
└── README.md                          # Project documentation and executive summary
```

---

### 7. Execution & Reproduction Guide

To run the data cleaning, splitting, and analytical EDA pipeline end-to-end:

```bash
# Clone the repository
git clone https://github.com/Samanzarei-9775/AAI500-Final-Project.git
cd AAI500-Final-Project

# Install required dependencies
pip install numpy pandas scikit-learn matplotlib seaborn statsmodels

# Launch Jupyter Notebook
jupyter notebook notebooks/data_cleaning_merged.ipynb
```

Execute all cells sequentially (`Kernel -> Restart & Run All`). The notebook executes in ~5 seconds, renders all figures inline, outputs statistical tables, and exports the six modeling partitions to the `Data/` directory.

---

### 8. Next Steps & Active Milestones

* [ ] Fit baseline models (Logistic Regression, Gaussian Naïve Bayes) and extract Odds Ratio summary tables.
* [ ] Train ensemble models (Random Forest, Gradient Boosting) and tune hyperparameters on `X_val`.
* [ ] Evaluate final test metrics (`Accuracy`, `Macro F1`, `One-vs-Rest ROC-AUC`) on `X_test`.
* [ ] Assemble Sections 1–6 of the PDF Technical Report and finalize the executed Jupyter Notebook appendix.
* [ ] Record the 8–10 minute executive presentation video with equal team distribution.
