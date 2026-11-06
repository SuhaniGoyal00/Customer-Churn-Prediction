# 📉 Customer Churn Prediction & Retention Intelligence

An end-to-end Machine Learning project that predicts customer churn, evaluates different ML models, and identifies customers who may be at risk of leaving a service.

Built using Python, Pandas, Scikit-learn, and Streamlit.

## 📌 Project Overview

Customer churn is an important business problem because losing customers affects revenue and growth. This project uses machine learning to analyse customer data, predict churn probability, and help businesses identify customers who may need retention strategies.

## 🛠️ Technologies Used

- **Python** – Data processing and model development
- **Pandas & NumPy** – Data cleaning and manipulation
- **Scikit-learn** – Machine Learning models
- **Streamlit** – Interactive dashboard
- **Matplotlib & Seaborn** – Data visualization
- **SHAP** – Model explainability (where supported)

## ✨ Key Features

- Customer churn analysis and visualization
- Data cleaning and preprocessing
- Comparison of Logistic Regression, Random Forest, and Extra Trees models
- Model evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Confusion matrix and ROC curve
- Individual customer churn prediction
- Low, Medium, and High churn-risk classification
- Batch customer prediction with downloadable CSV results
- Interactive dashboard with adjustable prediction threshold

## 🔄 Project Workflow

1. **Data Collection:** Load customer data or use the included demo dataset.
2. **Data Cleaning:** Handle missing values, duplicates, and data types.
3. **Exploratory Data Analysis:** Explore churn patterns based on customer tenure, contract type, and monthly charges.
4. **Data Preprocessing:** Apply encoding, scaling, and imputation.
5. **Model Training:** Train and compare multiple machine learning models.
6. **Model Evaluation:** Evaluate performance using classification metrics.
7. **Prediction:** Predict churn probability for individual and multiple customers.
8. **Visualization:** Display results through an interactive Streamlit dashboard.

## 📂 Project Structure

```text
churn-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── demo_customers.csv
│
├── src/
│   ├── data_pipeline.py
│   └── model_pipeline.py
│
├── models/
└── reports/
```

## ⚙️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the application**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 📊 Dataset

The project includes synthetic demo customer data so the application can run immediately. You can also upload a labelled Telco Customer Churn dataset for training and evaluation.

**Note:** Results from synthetic demo data are for demonstration purposes and should not be treated as real-world model performance.

## 🚀 Future Enhancements

- Deploy the dashboard online.
- Improve model performance using real-world datasets.
- Add advanced customer segmentation.
- Integrate additional explainability techniques.

## 👩‍💻 Author

**Suhani Goyal**  
B.E. Computer Science Engineering
