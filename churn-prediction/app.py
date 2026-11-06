from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from data_pipeline import generate_demo_data, clean_data, CATEGORICAL, NUMERIC, TARGET
from model_pipeline import train_models, save_package

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction & Retention Intelligence")
st.caption("End-to-end machine learning project · Data → Training → Evaluation → Explainable predictions")

@st.cache_data(show_spinner=False)
def default_data():
    path = Path("data/demo_customers.csv")
    if path.exists():
        return clean_data(pd.read_csv(path))
    return clean_data(generate_demo_data())

with st.sidebar:
    st.header("Project controls")
    uploaded = st.file_uploader("Upload labelled customer CSV", type=["csv"])
    st.caption("Training CSV must include a Churn column (Yes/No). The app also accepts the standard Telco Customer Churn format.")
    tune = st.checkbox("Tune Logistic Regression with GridSearchCV", value=False)
    retrain = st.button("Train / retrain models", type="primary")
    st.divider()
    st.caption("Demo mode uses generated synthetic data. Replace it with the real Telco dataset for a valid academic evaluation.")

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else default_data()
    df = clean_data(raw)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if "package" not in st.session_state or retrain or uploaded is not None:
    with st.spinner("Cleaning data and training models..."):
        try:
            st.session_state.package = train_models(df, tune=tune)
            save_package(st.session_state.package)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()
package = st.session_state.package
models, metrics, best_name = package["models"], package["metrics"], package["best_name"]

pages = st.tabs(["Overview & EDA", "Model Evaluation", "Single Customer", "Batch Prediction", "Data & Methodology"])
with pages[0]:
    st.subheader("Customer portfolio")
    a,b,c,d = st.columns(4)
    churn_rate = (df[TARGET] == "Yes").mean()
    a.metric("Customers", f"{len(df):,}")
    b.metric("Churn rate", f"{churn_rate:.1%}")
    c.metric("Average monthly charge", f"${df['MonthlyCharges'].mean():,.2f}")
    d.metric("Selected model", best_name)
    left,right = st.columns(2)
    with left:
        st.markdown("**Churn distribution**")
        st.bar_chart(df[TARGET].value_counts().rename_axis("Churn").to_frame("Customers"))
    with right:
        st.markdown("**Churn rate by contract**")
        if "Contract" in df:
            st.bar_chart(df.groupby("Contract")[TARGET].apply(lambda s: (s == "Yes").mean()).rename("Churn rate"))
    left,right = st.columns(2)
    with left:
        st.markdown("**Churn rate by tenure group**")
        tenure_group = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0–12", "13–24", "25–48", "49–72"])
        st.bar_chart(df.assign(TenureGroup=tenure_group).groupby("TenureGroup", observed=False)[TARGET].apply(lambda s: (s == "Yes").mean()).rename("Churn rate"))
    with right:
        st.markdown("**Monthly charges by churn status**")
        st.bar_chart(df.groupby(TARGET)["MonthlyCharges"].mean().rename("Average monthly charges"))
    st.info("Business use: identify segments with elevated churn and prioritize retention outreach. These associations do not prove causation.")

with pages[1]:
    st.subheader("Model comparison on the held-out test set")
    st.dataframe(metrics.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}), use_container_width=True, hide_index=True)
    chosen = st.selectbox("Inspect model", list(models.keys()), index=list(models.keys()).index(best_name))
    threshold = st.slider("Churn alert threshold", 0.10, 0.90, 0.50, 0.05, help="Lower thresholds usually catch more churners but can create more false alarms.")
    y_true = package["y_test"].to_numpy()
    probs = package["predictions"][chosen]["prob"]
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0,1])
    l,r = st.columns(2)
    with l:
        st.markdown("**Confusion matrix** (rows = actual No/Yes; columns = predicted No/Yes)")
        st.dataframe(pd.DataFrame(cm, index=["Actual No churn", "Actual Churn"], columns=["Predicted No churn", "Predicted Churn"]), use_container_width=True)
    with r:
        st.markdown("**ROC curve**")
        fpr,tpr,_ = roc_curve(y_true, probs)
        fig,ax = plt.subplots()
        ax.plot(fpr,tpr,label=f"AUC = {roc_auc_score(y_true, probs):.3f}")
        ax.plot([0,1],[0,1],linestyle="--")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=chosen)
        ax.legend(); st.pyplot(fig); plt.close(fig)
    st.caption(f"At threshold {threshold:.2f}: recall = {(cm[1,1]/max(1,cm[1].sum())):.3f}; precision = {(cm[1,1]/max(1,cm[:,1].sum())):.3f}. Threshold choice should reflect retention-team capacity and costs.")
    st.markdown("**Global feature importance**")
    try:
        fitted = models[chosen]
        names = fitted.named_steps["prep"].get_feature_names_out()
        estimator = fitted.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            values = np.abs(estimator.coef_[0])
        else:
            values = None
        if values is not None:
            imp = pd.Series(values, index=names).sort_values().tail(15)
            st.bar_chart(imp)
    except Exception as e:
        st.caption(f"Feature importance could not be displayed: {e}")
    st.caption("Importance indicates model reliance/association, not proof that changing a feature will prevent churn.")

with pages[2]:
    st.subheader("Predict churn for one customer")
    st.write("Enter a customer's service and billing details. Missing fields use the same preprocessing pipeline as training.")
    with st.form("single_customer"):
        col1,col2,col3 = st.columns(3)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly charges", min_value=0.0, max_value=1000.0, value=75.0)
            total = st.number_input("Total charges", min_value=0.0, max_value=100000.0, value=float(tenure*75))
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0,1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone = st.selectbox("Phone service", ["Yes", "No"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        with col3:
            security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
            device = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
            tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        submitted = st.form_submit_button("Predict churn risk", type="primary")
    if submitted:
        customer = {"gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,"tenure":tenure,
                    "PhoneService":phone,"MultipleLines":"Unknown","InternetService":internet,"OnlineSecurity":security,
                    "OnlineBackup":backup,"DeviceProtection":device,"TechSupport":tech,"StreamingTV":tv,"StreamingMovies":movies,
                    "Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total}
        row = pd.DataFrame([customer]).reindex(columns=package["feature_columns"])
        probability = float(models[best_name].predict_proba(row)[0,1])
        category = "High" if probability >= .60 else "Medium" if probability >= .30 else "Low"
        st.metric("Estimated churn probability", f"{probability:.1%}")
        st.progress(probability)
        st.subheader(f"Risk category: {category}")
        st.write("**Potential risk signals to review:**")
        reasons=[]
        if contract == "Month-to-month": reasons.append("Month-to-month contract: customers have less commitment than longer contracts.")
        if tenure < 12: reasons.append("Short tenure: newer customers may still be deciding whether to stay.")
        if payment == "Electronic check": reasons.append("Electronic-check payment is often associated with higher churn in Telco datasets.")
        if tech == "No": reasons.append("No tech support selected: check whether support needs are being met.")
        if monthly >= 85: reasons.append("Higher monthly charges: review plan fit and perceived value.")
        if not reasons: reasons.append("No simple rule-based risk signal was triggered; use the model score as a screening signal, not a certainty.")
        for reason in reasons: st.write("• " + reason)
        try:
            import shap
            fitted = models[best_name]
            estimator = fitted.named_steps["model"]
            transformed = fitted.named_steps["prep"].transform(row)
            if hasattr(estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(transformed)
                if isinstance(shap_values, list):
                    local_values = np.asarray(shap_values[-1]).reshape(-1)
                else:
                    arr = np.asarray(shap_values)
                    if arr.ndim == 3:
                        local_values = arr[0, :, -1]
                    elif arr.ndim == 2:
                        local_values = arr[0]
                    else:
                        local_values = arr.reshape(-1)
                feature_names = fitted.named_steps["prep"].get_feature_names_out()
                local = pd.Series(local_values, index=feature_names).sort_values(key=np.abs, ascending=False).head(8)
                st.markdown("**Local SHAP feature contributions**")
                st.caption("Positive values push the model toward churn; negative values push away from churn. SHAP output format can vary by model/library version.")
                st.bar_chart(local)
            elif hasattr(estimator, "coef_"):
                st.caption("For Logistic Regression, inspect the global coefficient chart in Model Evaluation; install/use a compatible SHAP version for local attributions.")
        except Exception as e:
            st.caption(f"SHAP explanation unavailable in this environment: {e}")
        st.caption("A probability is a model estimate, not certainty. Review explanations with real data before operational use.")

with pages[3]:
    st.subheader("Batch prediction and CSV export")
    batch_file = st.file_uploader("Upload customers to score (CSV without Churn is okay)", type=["csv"], key="batch")
    if batch_file is not None:
        try:
            batch = pd.read_csv(batch_file)
            ids = batch["customerID"].astype(str) if "customerID" in batch else pd.Series([f"UPLOAD{i+1:05d}" for i in range(len(batch))])
            cleaned = batch.copy()
            for col in package["feature_columns"]:
                if col not in cleaned: cleaned[col] = np.nan
            X_batch = cleaned[package["feature_columns"]].copy()
            for col in CATEGORICAL:
                if col in X_batch: X_batch[col] = X_batch[col].fillna("Unknown").astype(str)
            prob = models[best_name].predict_proba(X_batch)[:,1]
            output = pd.DataFrame({"customerID":ids, "churn_probability":prob,
                                   "risk_category":np.where(prob>=.60,"High",np.where(prob>=.30,"Medium","Low")),
                                   "prediction":np.where(prob>=.5,"Churn","No Churn")})
            st.dataframe(output.head(100), use_container_width=True)
            st.download_button("Download predictions as CSV", output.to_csv(index=False).encode("utf-8"), "churn_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not score this CSV: {e}")
    else:
        st.info("Upload a customer CSV to score many customers at once. For prediction files, the Churn label is not required.")

with pages[4]:
    st.subheader("Data quality report")
    q1,q2,q3 = st.columns(3)
    q1.metric("Duplicate rows removed", int(len(raw)-len(df)))
    q2.metric("Missing cells remaining", int(df.isna().sum().sum()))
    q3.metric("Features used", len(package["feature_columns"]))
    st.dataframe(pd.DataFrame({"Column":df.columns,"Data type":df.dtypes.astype(str).values,"Missing values":df.isna().sum().values,"Unique values":df.nunique().values}), use_container_width=True, hide_index=True)
    st.markdown("### Methodology")
    st.markdown("""
    1. **Problem:** identify customers at risk of cancelling.
    2. **Cleaning:** standardize labels, convert numeric columns, remove duplicate rows, and handle missing values inside the training pipeline.
    3. **Split:** stratified 80/20 train/test split with a fixed seed.
    4. **Preprocessing:** median-impute and scale numeric fields; most-frequent-impute and one-hot encode categorical fields.
    5. **Imbalance:** class weights are used so churners receive extra training emphasis; no synthetic samples are created in the test set.
    6. **Models:** Logistic Regression, Random Forest, and Extra Trees are compared.
    7. **Evaluation:** held-out accuracy, precision, recall, F1 and ROC-AUC; confusion matrix and ROC curve.
    8. **Selection:** choose by recall first, then ROC-AUC and F1. A production decision should also consider precision, outreach cost and capacity.
    9. **Deployment:** Streamlit app; trained model package is saved under `models/`.
    """)
    st.warning("If you are using generated demo data, the metrics only show that the pipeline runs. They are not evidence of real-world performance. Train and evaluate with the actual Telco CSV before presenting results.")
    st.download_button("Download cleaned dataset", df.to_csv(index=False).encode("utf-8"), "cleaned_customers.csv", "text/csv")
