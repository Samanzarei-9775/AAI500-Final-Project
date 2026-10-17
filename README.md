# Predicting Workforce Volatility: An Executive Framework for AI-Driven Layoff Risk

**University of San Diego — Master of Science in Applied Artificial Intelligence (MS-AAI)**  
**Course:** AAI-500: Probability and Statistics for Artificial Intelligence  
**Final Project — Team 7**

---

## Project Overview

### 1. Project Scope & Research Question

* **Title:** *Predicting Workforce Volatility: An Executive Framework for AI-Driven Layoff Risk*
* **Core Research Question:** *"What is an employee's risk of layoff based on their industry sector, career tenure, routine task percentage, tasks automated percentage, and AI tooling adoption?"*
* **Target Variable ($y$):** `Layoff_Risk` modeled as an ordinal multi-class outcome:
  * `0`: **Low Risk**
  * `1`: **Medium Risk**
  * `2`: **High Risk**
* **Context & Framing:** Utilizes a structured 20,000-record dataset framed as an **HR decision-support simulation** to evaluate employee vulnerability and protective career factors during enterprise AI adoption.

---

### 2. Feature Architecture

Transitioning from demographic indicators to workload- and automation-centric features provides direct empirical separation across risk tiers:

* **Selected Predictors ($X$ $\to$ 11 Model Features):**
  * `Industry`: Nominal categorical feature one-hot encoded with `drop_first=True`, explicitly designating **`IT`** as the dropped reference baseline to anchor regression odds ratios ($\exp(\beta)$). Generates 7 indicator columns representing contrasts against the tech benchmark:
    * `Industry_Education`, `Industry_Finance`, `Industry_Healthcare`, `Industry_Logistics`, `Industry_Manufacturing`, `Industry_Retail`, `Industry_Telecom`
  * `Years_of_Experience`: Continuous numerical variable ($z$-score standardized).
  * `Routine_Task_Percentage`: Continuous percentage variable [0, 100] capturing task repetitiveness ($z$-score standardized).
  * `Tasks_Automated_Percentage`: Continuous percentage variable [0, 100] measuring existing workflow automation ($z$-score standardized).
  * `Number_of_AI_Tools_Used`: Discrete integer count [0, 10] reflecting employee-level AI integration ($z$-score standardized).

---

### 3. Data Preprocessing & Validation Pipeline

* **Integrity & Sanity Audits:**
  * Deduplication verification confirms zero redundant employee records across the 20,000 observations.
  * Career tenure domain sanity constraint enforced: $(\text{Age} - \text{Years of Experience}) \ge 16$, validating that no career history reflects commencement prior to standard legal working age.
  * Comprehensive null audit confirms zero missing values across all columns.
  * Numeric range audit confirms all percentage predictors lie strictly within [0, 100] and counts are non-negative.
* **Stratified 2-Way Holdout Data Partitioning (80/20 Split):**
  * **Training Set:** 80% ($N = 16,000$ records $\times$ 11 predictors)
  * **Testing Set:** 20% ($N = 4,000$ records $\times$ 11 predictors)
  * Stratification preserves identical target class distribution across both partitions (~33.0% Low, ~33.0% Medium, ~34.0% High).
* **Leakage-Free Feature Scaling:**
  * `StandardScaler` is fitted **strictly on `X_train`** ($\mu_{\text{train}}, \sigma_{\text{train}}$) and applied downstream to transform `X_test`, maintaining complete out-of-sample data isolation. One-hot dummy columns remain intact as unscaled binary indicators.
* **Processed Data Artifacts:**
  * Partitioned and standardized modeling sets are exported to the [`Data/`](file:///c:/Users/19496/OneDrive/Desktop/MSAAI/AntiGravity%20Final%20Proejct/AAI500-Final-Project/Data) directory:
    * `processed_X_train.csv` (16,000 $\times$ 11) & `processed_y_train.csv` (16,000 $\times$ 1)
    * `processed_X_test.csv` (4,000 $\times$ 11) & `processed_y_test.csv` (4,000 $\times$ 1)

---

### 4. Analytical Exploratory Data Analysis (EDA) Highlights

Exploratory analyses conducted strictly on the training partition ($N = 16,000$, zero data leakage) reveal four core empirical takeaways:

1. **Routine Work and Automation Exposure Dominate Risk Separation:**
   * Median **routine task share** climbs sharply from **26.0%** (Low Risk) to **50.0%** (Medium) and **77.0%** (High).
   * Median **tasks automated** follows the same trajectory (**17.0% $\to$ 34.0% $\to$ 55.0%**). These two features provide near-complete separation between the extreme risk tiers.
2. **AI Tooling Adoption Threshold Effect:**
   * Workers utilizing **0–1 AI tools** are ~56% Low Risk and <10% High Risk.
   * At **3 tools**, High Risk jumps abruptly to **58.6%**, expanding to **74.9% at 6 tools** and **83.7% at 10 tools**. The non-linear step-function strongly benefits non-parametric tree ensembles.
3. **Sector Volatility Disparity Relative to Baseline IT:**
   * Operational and supply-chain intensive sectors exhibit elevated risk: **Manufacturing (48.2% High Risk)**, **Retail (41.7%)**, and **Logistics (40.9%)**.
   * Essential human services demonstrate superior stability: **Education (19.9% High Risk)** and **Healthcare (21.0% High Risk)**.
   * The reference baseline, **IT (27.8% High Risk)**, occupies an optimal intermediate position to capture both risk premiums and discounts.
4. **Career Tenure Buffering & Multicollinearity Diagnostics:**
   * Career tenure provides a modest protective buffer: median experience declines from **8.0 years** (Low Risk) to **7.0 years** (Medium) and **6.0 years** (High).
   * `Routine_Task_Percentage` and `Tasks_Automated_Percentage` are strongly correlated ($r = 0.889$, Variance Inflation Factors $\text{VIF} \approx 5.6$ and $4.8$). In tree ensembles, splits are shared between them; in Naïve Bayes, this tests the conditional independence assumption.

---

### 5. Benchmarking Model Suite & Test Performance

A 4-model suite balancing statistical inference, interpretability, and predictive performance evaluated on the holdout **Test Set ($N = 4,000$, 20%)**:

| Model | Test Accuracy | Macro Sensitivity (Recall) | Macro Specificity (TNR) | Generalization Gap (Train − Test) | Primary Role & Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gradient Boosting (GB)** | **74.95%** | **0.7486** | **0.8748** | **+1.33%** | **Top Predictor:** Captures non-linear thresholds; high sensitivity on extreme tiers (83.5% High, 80.0% Low). |
| **Random Forest (RF)** | **74.88%** | **0.7482** | **0.8744** | **+2.45%** | **Bagging Ensemble:** Robust non-parametric baseline with Gini feature importance validation. |
| **Gaussian Naïve Bayes (NB)** | **73.35%** | **0.7328** | **0.8669** | **−1.04%** | **Probabilistic Baseline:** Fast generative classifier; highest Low-Risk recall (**86.59%**). |
| **Multinomial Logistic Regression (LR)** | **71.70%** | **0.7158** | **0.8585** | **+0.10%** | **Parametric Inference:** Provides interpretable parameter coefficients ($\beta$) and Odds Ratios ($e^\beta$) relative to IT. |

*Majority-Class Baseline Accuracy on Test Set:* **34.00%**. All four models outperform the naive baseline by **37–41 percentage points**.

---

### 6. Repository Architecture

```text
AAI500-Final-Project/
│
├── Data/
│   ├── Raw-AI-Dataset.csv                 # Raw 20,000-record dataset
│   ├── processed_X_train.csv              # Scaled training predictors (16,000 x 11)
│   ├── processed_X_test.csv               # Scaled holdout test predictors (4,000 x 11)
│   ├── processed_y_train.csv              # Training target labels (16,000 x 1)
│   └── processed_y_test.csv               # Holdout test target labels (4,000 x 1)
│
├── notebooks/
│   ├── data_cleaning_merged.ipynb         # Master data cleaning, encoding, 80/20 splitting, and Section 3 EDA
│   ├── Training_models_merged.ipynb       # Master multi-model notebook benchmarking LR, RF, GB, and NB
│   ├── data_training_Mustafa_GB.ipynb     # Dedicated Gradient Boosting model training & evaluation (Mustafa)
│   ├── data_training_Mustafa_NB.ipynb     # Dedicated Gaussian Naïve Bayes model training & evaluation (Mustafa)
│   ├── data_training_Saman_LR.ipynb       # Dedicated Multinomial Logistic Regression notebook (Saman)
│   ├── data_training_Saman_RF.ipynb       # Dedicated Random Forest classifier notebook (Saman)
│   ├── Data-Cleaning.ipynb                # Team data cleaning reference notebook
│   └── data_cleaning_Saman_Version.ipynb  # Alternative cleaning reference notebook
│
├── .gitignore                             # Ignores virtual environments (.venv) and checkpoints
└── README.md                              # Comprehensive project documentation and executive summary
```

---

### 7. Execution & Reproduction Guide

To reproduce the data cleaning and model benchmarking pipeline end-to-end:

```bash
# 1. Clone the repository
git clone https://github.com/Samanzarei-9775/AAI500-Final-Project.git
cd AAI500-Final-Project

# 2. Set up virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # On Linux/macOS
# .\.venv\Scripts\Activate.ps1   # On Windows PowerShell

pip install numpy pandas scikit-learn matplotlib seaborn statsmodels jupyter

# 3. Execute master cleaning and splitting pipeline
jupyter nbconvert --to notebook --execute --inplace notebooks/data_cleaning_merged.ipynb

# 4. Execute master multi-model training pipeline
jupyter nbconvert --to notebook --execute --inplace notebooks/Training_models_merged.ipynb
```

Each notebook includes an automated self-installing environment setup cell at index 0 to ensure immediate execution in any Jupyter/VS Code kernel.

---

### 8. Project Milestones & Deliverables Status

* [x] Audit raw dataset integrity (duplicates, nulls, tenure domain constraint).
* [x] Implement 5-feature architecture (`Industry`, `Years_of_Experience`, `Routine_Task_Percentage`, `Tasks_Automated_Percentage`, `Number_of_AI_Tools_Used`).
* [x] Establish zero-leakage 80/20 stratified partitioning ($N = 16,000$ train / $N = 4,000$ test).
* [x] Conduct analytical EDA strictly on the training partition.
* [x] Train and benchmark full 4-model suite (Logistic Regression, Random Forest, Gradient Boosting, Gaussian Naïve Bayes).
* [x] Consolidate multi-model training into [`Training_models_merged.ipynb`](file:///c:/Users/19496/OneDrive/Desktop/MSAAI/AntiGravity%20Final%20Proejct/AAI500-Final-Project/notebooks/Training_models_merged.ipynb).
* [ ] Finalize executive presentation slides and recording.
* [ ] Compile final academic PDF technical report.
