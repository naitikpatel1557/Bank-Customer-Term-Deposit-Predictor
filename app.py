#! C:\Users\Asus\Bank_Cust_Deposit\venv\Scripts\python.exe
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('bank_model.pkl')

model = load_model()

st.title("Bank Term Deposit Predictor")
st.markdown("Predict whether a customer will subscribe to a term deposit based on campaign data and socioeconomic indicators.")

# Interactive Threshold Tuner
st.sidebar.header("Decision Settings")
threshold = st.sidebar.slider("Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.sidebar.markdown("""
*Lowering the threshold catches more potential subscribers, but increases the number of false positives (calling people who aren't interested).*
""")

# Input Form
st.header("Customer Information")
with st.form("customer_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
        education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'])
        default = st.selectbox("Has Credit in Default?", ['no', 'yes', 'unknown'])
        
    with col2:
        housing = st.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])
        contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
        month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        day_of_week = st.selectbox("Last Contact Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    
    st.subheader("Campaign History")
    col3, col4 = st.columns(2)
    with col3:
        campaign = st.number_input("Contacts during this campaign", min_value=1, value=1)
        pdays = st.number_input("Days since last contact (999 = never)", value=999)
    with col4:
        previous = st.number_input("Contacts before this campaign", min_value=0, value=0)
        poutcome = st.selectbox("Previous Campaign Outcome", ['failure', 'nonexistent', 'success'])
        
    st.subheader("Socioeconomic Indicators")
    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.994)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857)
    nr_employed = st.number_input("Number of Employees", value=5191.0)
    
    submitted = st.form_submit_button("Predict Subscription")

if submitted:
    # Create input dataframe ensuring column names match training data exactly
    input_data = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }])
    
    # Get Probability
    prob = model.predict_proba(input_data)[0][1]
    
    # Apply custom threshold
    prediction = "Yes (Will Subscribe)" if prob >= threshold else "No (Will Not Subscribe)"
    
    st.divider()
    st.subheader("Prediction Results")
    
    if prob >= threshold:
        st.success(f"**Prediction: {prediction}**")
    else:
        st.error(f"**Prediction: {prediction}**")
        
    st.metric(label="Probability of Subscribing", value=f"{prob:.1%}")