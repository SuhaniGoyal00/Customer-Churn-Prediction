"""CSV loading, cleaning, feature preparation, and demo-data generation."""
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
ALIASES = {"customerid": "customerID", "monthlycharges": "MonthlyCharges",
           "totalcharges": "TotalCharges", "seniorcitizen": "SeniorCitizen"}


def generate_demo_data(n=2500, seed=42):
    """Generate clearly labelled demo data; not a replacement for real business data."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.35, .45, .20])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n)
    monthly = np.round(rng.uniform(20, 120, n), 2)
    senior = rng.choice([0, 1], n, p=[.84, .16])
    partner = rng.choice(["Yes", "No"], n)
    dependents = rng.choice(["Yes", "No"], n, p=[.3, .7])
    tech = rng.choice(["Yes", "No", "No internet service"], n, p=[.25, .55, .20])
    logit = (-1.25 + (contract == "Month-to-month") * 1.25 + (contract == "Two year") * -1.0
             + (tenure < 12) * 1.1 + (tenure > 48) * -.75 + (internet == "Fiber optic") * .35
             + (payment == "Electronic check") * .45 + (tech == "No") * .35 + senior * .25
             + (monthly > 85) * .25 + rng.normal(0, .65, n))
    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob, "Yes", "No")
    df = pd.DataFrame({
        "customerID": [f"DEMO{i:05d}" for i in range(1, n+1)], "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n), "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": internet, "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n), "TechSupport": tech,
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n), "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n), "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": np.round(monthly * tenure + rng.normal(0, 30, n), 2), "Churn": churn
    })
    df["TotalCharges"] = df["TotalCharges"].clip(lower=0)
    return df


def clean_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {c: ALIASES[c.lower()] for c in df.columns if c.lower() in ALIASES and c != ALIASES[c.lower()]}
    df = df.rename(columns=rename)
    if TARGET not in df.columns:
        raise ValueError("CSV must contain a 'Churn' target column with Yes/No labels for training.")
    if "tenure" not in df.columns or "MonthlyCharges" not in df.columns:
        raise ValueError("CSV must include at least 'tenure' and 'MonthlyCharges'. Use the Telco Customer Churn dataset format.")
    df = df.drop_duplicates().reset_index(drop=True)
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title().replace({"1": "Yes", "0": "No", "True": "Yes", "False": "No"})
    df = df[df[TARGET].isin(["Yes", "No"])].copy()
    if df.empty:
        raise ValueError("No valid Yes/No values found in Churn column.")
    for col in CATEGORICAL:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].astype("string").fillna("Unknown").str.strip().replace({"": "Unknown", "nan": "Unknown", "?": "Unknown"})
    if "customerID" not in df.columns:
        df.insert(0, "customerID", [f"CUST{i:05d}" for i in range(1, len(df)+1)])
    return df.reset_index(drop=True)
