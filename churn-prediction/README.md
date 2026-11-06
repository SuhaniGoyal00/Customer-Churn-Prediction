# Customer Churn Prediction & Retention Intelligence

An end-to-end machine-learning project for exploring customer churn, training and comparing models, evaluating performance, predicting individual customers, and scoring batch customer files. Built with Python, pandas, scikit-learn, and Streamlit.

> **Important:** The included data generator creates synthetic demo data so the app can run immediately. Its scores are not evidence of real-world performance. For a credible report, download the Telco Customer Churn dataset from Kaggle and train/evaluate on that dataset.

## Features

- CSV upload and basic data cleaning (type conversion, missing-value normalization, duplicate removal)
- EDA: overall churn, churn by contract and tenure, monthly charge comparison
- Leakage-aware stratified 80/20 train-test split
- Preprocessing inside an sklearn pipeline: imputation, scaling and one-hot encoding
- Logistic Regression, Random Forest, Extra Trees and XGBoost (when installed) model comparison
- Class imbalance support through class weights (SMOTE is not used)
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve and adjustable alert threshold
- Single-customer probability prediction with low/medium/high risk bands
- Batch predictions and downloadable CSV
- Saved model package in `models/churn_models.joblib`

## Requirements

- Python 3.10 or newer recommended
- Internet only to install Python packages (not required after installation)

## Run on Windows (VS Code)

1. Extract the ZIP file.
2. Open the `churn-prediction` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or run `.venv\Scripts\python.exe -m pip` directly.
5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Start the app:

   ```powershell
   streamlit run app.py
   ```
7. The browser opens a local URL, usually `http://localhost:8501`.

## Use the real Telco dataset

Download the Telco Customer Churn CSV from Kaggle. In the app sidebar, upload the labelled CSV for training. The training file must have a `Churn` column containing `Yes` and `No`, plus at least `tenure` and `MonthlyCharges`. The standard Telco dataset columns are supported. `TotalCharges` values that contain spaces are converted to missing values and imputed by the pipeline.

For batch scoring, upload a customer CSV with the same feature columns but **without** the Churn label. Missing optional features are filled with safe placeholders, but prediction quality is best when input columns match the training schema.

## Project structure

```text
churn-prediction/
├── app.py                  # Streamlit interface and user workflows
├── requirements.txt        # Python dependencies
├── README.md               # Setup, methodology and project guide
├── data/
│   └── demo_customers.csv  # Synthetic data to make first run easy
├── models/                 # Saved model package after training
├── reports/                # Place final report and figures here
└── src/
    ├── data_pipeline.py    # Demo data, schema normalization and cleaning
    └── model_pipeline.py   # preprocessing, training, comparison and saving
```

## How the ML pipeline works

1. Split labelled data into 80% train and 20% test with stratification.
2. Fit preprocessing only on the training fold, avoiding test-set leakage.
3. One-hot encode categorical features; impute and scale numeric features.
4. Train class-weighted Logistic Regression, Random Forest and Extra Trees.
5. Compare models on the untouched test split. The app selects the model with highest recall, with ROC-AUC and F1 as tie-breakers.
6. Score a customer using `predict_proba`; map the probability to risk bands: Low <30%, Medium 30–60%, High ≥60%.
7. Export batch scores for a retention team to review.

## Metrics in plain English

- **Accuracy:** fraction of all predictions that are correct.
- **Precision:** among customers flagged as churners, the fraction who actually churned.
- **Recall:** among actual churners, the fraction the model catches. This project emphasizes recall to avoid missing customers who may leave.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** how well the model ranks churners above non-churners across thresholds.
- **Confusion matrix:** counts of true negatives, false positives, false negatives and true positives.

## Limitations and responsible use

- Predictions are estimates, not certainty; a risk score does not establish why a customer leaves.
- The app's single-customer reasons are transparent rule-based hints, **not SHAP values**. The feature-importance chart is global model importance/coefficient magnitude, not an individual SHAP explanation.
- XGBoost is included when its package is installed. SHAP local contributions are attempted for tree-based models; library/model-version differences may make explanations unavailable, in which case the app displays a clear fallback note.
- Synthetic demo data should never be used to claim real-world model performance.
- Review false positives, fairness, privacy and the cost/capacity of retention offers before operational use.

## Suggested team split (4 members)

1. **Data Analyst / Data Engineer:** dataset sourcing, data quality, EDA and charts.
2. **ML Engineer:** train-test split, preprocessing, models, class imbalance and evaluation.
3. **Testing & Explainability Engineer:** test cases, threshold analysis, error analysis, and a validated SHAP integration as an extension.
4. **App Developer / Documentation Lead:** Streamlit app, CSV upload/download, README, report and presentation.

## Suggested test checklist

- CSV with both churn classes trains successfully.
- Missing `TotalCharges` values do not crash training.
- Invalid/missing `Churn` column produces a readable error.
- Batch input with unseen categories still predicts because one-hot encoding ignores unknown categories.
- Changing the threshold changes the confusion matrix.
- Downloaded batch CSV contains ID, probability, risk band and prediction.
- Final report records the dataset version, split strategy, chosen model and all test metrics.


# CUSTOMER CHURN PREDICTION PROJECT
## Terminal Commands – Every Time You Run the Project

**Project location:** `C:\Users\Suhani\Downloads\customer-churn-prediction-project\churn-prediction`

### Step 1 – Open project folder
Open VS Code → Open Folder → Select `churn-prediction`.

### Step 2 – Open terminal
Press `Ctrl + Shift + ~` or select Terminal → New Terminal.

### Step 3 – Navigate to project folder

```powershell
cd "$HOME\Downloads\customer-churn-prediction-project\churn-prediction"
```

### Step 4 – Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5 – Run the project

```powershell
streamlit run app.py
```

### Step 6 – Open dashboard
Your browser should open automatically.

If not, visit:

`http://localhost:8501`

### To stop the project
Press `Ctrl + C` in the terminal.

---

## IMPORTANT

- Do NOT create a new virtual environment every time.
- Do NOT install requirements every time.
- Do NOT delete the `.venv` folder.
- Activate the environment whenever you open a new terminal.
- Run `streamlit run app.py` whenever you want to launch the dashboard.

**These are your everyday commands.**