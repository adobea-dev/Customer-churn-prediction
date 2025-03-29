import streamlit as st
import pandas as pd
import pickle

# Load the pruned decision tree model
with open("pruned_decision_tree.pkl", "rb") as file:
    model = pickle.load(file)

# App Title
st.title("Customer Churn Prediction App")
st.markdown("Enter customer details below to predict whether they are likely to churn.")

# Collect user input features
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
usage_frequency = st.slider("Usage Frequency (times per month)", min_value=0, max_value=30, value=10)
support_calls = st.number_input("Number of Support Calls", min_value=0, max_value=50, value=2)
payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=90, value=5)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Annual", "Quartely"])
total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=20000.0, value=500.0)
last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=365, value=30)
age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "56+"])

# Encode categorical variables
gender_map = {"Male": 0, "Female": 1}
subscription_map = {"Basic": 0, "Standard": 1, "Premium": 2}
contract_map = {"Monthly": 0, "Annual": 1, "Quartely": 2}
age_group_map = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "56+": 4}

# Convert user inputs to numerical values
gender = gender_map[gender]
subscription_type = subscription_map[subscription_type]
contract_length = contract_map[contract_length]
age_group = age_group_map[age_group]

# Create DataFrame with input data
input_data = pd.DataFrame([[gender, tenure, usage_frequency, support_calls, 
                            payment_delay, subscription_type, contract_length, 
                            total_spend, last_interaction, age_group]],
                          columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

# Add missing columns with default values (adjust based on training data)
for missing_col in ["10", "11", "12", "13"]:
    input_data[missing_col] = 0  # Replace 0 with actual default values if known

# Make a prediction
prediction = model.predict(input_data)

# Convert prediction output to human-readable format
churn_result = "Churn" if prediction[0] == 1 else "No Churn"

# Display the result
st.subheader("Prediction Result:")
st.write(f"The customer is likely to: **{churn_result}**")