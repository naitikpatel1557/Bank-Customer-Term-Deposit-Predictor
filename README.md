# Bank Customer Term Deposit Predictor 🏦📊

A complete machine learning pipeline and interactive web application built to predict whether a bank customer will subscribe to a term deposit based on historical campaign data, demographics, and macroeconomic indicators. 

Developed as part of an internship initiative at Brainybeam Technology.

## 📌 Project Overview
Bank telemarketing campaigns are expensive and time-consuming. Calling every customer is inefficient. This project solves that business problem by using artificial intelligence to identify high-probability leads *before* a call is made. By focusing only on customers likely to convert, financial institutions can optimize their marketing budgets, save time, and increase overall conversion rates.

## ✨ Key Features
* **End-to-End ML Pipeline:** Automated handling of categorical text data (One-Hot Encoding) seamlessly integrated with the classifier.
* **XGBoost Algorithm:** Utilizes Extreme Gradient Boosting (`XGBClassifier`) for highly accurate predictions on tabular data.
* **Interactive UI:** A clean, user-friendly frontend built with Streamlit, allowing non-technical stakeholders to input customer data and get real-time predictions.
* **Dynamic Probability Tuner:** Includes an interactive threshold slider, enabling business managers to adjust the model's aggressiveness based on daily marketing capacity (e.g., catching more leads vs. prioritizing only the most certain conversions).

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn, XGBoost
* **Model Serialization:** Joblib
* **Frontend Web App:** Streamlit

## 📂 Dataset
This project uses the **Bank Marketing Dataset** (originally from the UCI Machine Learning Repository).
* **Target Variable:** `deposit` (Binary: 'yes' or 'no')
* **Input Features:** Customer demographics (age, job, marital status), past campaign history, and real-time economic indicators.
* *Note: The `duration` column is dropped during training to prevent data leakage, as call duration is unknown prior to making the call.*

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/naitikpatel1557/bank-term-deposit-predictor.git](https://github.com/naitikpatel1557/bank-term-deposit-predictor.git)
   cd bank-term-deposit-predictor
