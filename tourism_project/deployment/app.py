import streamlit as st
import joblib
import pandas as pd

from huggingface_hub import hf_hub_download

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="jbalabantaray/tourism-model", filename="best_tourism_model_v1.joblib")

# Load the preprocessor and the trained model
try:
    best_model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'best_tourism_model_v1.joblib' is in the correct directory.")
    st.stop()

st.title("Tourism Package Prediction")
st.write("Enter the details below to predict if a customer will take the tourism package.")

# Define input widgets for each feature
# Based on the previous data analysis, we'll create appropriate input fields.
# Assuming the order and names of features are consistent with training data

# Example input widgets (adjust based on actual data types and ranges)
unnamed_0 = st.number_input("Unnamed: 0", min_value=0, step=1)
customer_id = st.number_input("CustomerID", min_value=200000, step=1)
age = st.number_input("Age", min_value=18, max_value=61, step=1)
typeofcontact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
citytier = st.number_input("City Tier", min_value=1, max_value=3, step=1)
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=5.0, max_value=127.0, step=0.1)
occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
gender = st.selectbox("Gender", ['Female', 'Male', 'Fe Male'])
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, step=1)
numberoffollowups = st.number_input("Number of Followups", min_value=1.0, max_value=6.0, step=1.0)
productpitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
preferredpropertystar = st.number_input("Preferred Property Star", min_value=3.0, max_value=5.0, step=1.0)
maritalstatus = st.selectbox("Marital Status", ['Single', 'Divorced', 'Married', 'Unmarried'])
numberoftrips = st.number_input("Number of Trips", min_value=1.0, max_value=22.0, step=1.0)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, step=1)
owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=3.0, step=1.0)
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
monthlyincome = st.number_input("Monthly Income", min_value=1000.0, max_value=98678.0, step=100.0)


if st.button("Predict"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([[
        unnamed_0, customer_id, age, typeofcontact, citytier, durationofpitch,
        occupation, gender, numberofpersonvisiting, numberoffollowups,
        productpitched, preferredpropertystar, maritalstatus, numberoftrips,
        passport, pitchsatisfactionscore, owncar, numberofchildrenvisiting,
        designation, monthlyincome
    ]], columns=[
        'Unnamed: 0', 'CustomerID', 'Age', 'TypeofContact', 'CityTier',
        'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
        'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
    ])

    # Make prediction
    prediction = best_model.predict(input_data)

    # Display result
    if prediction[0] == 1:
        st.success("Prediction: Customer is likely to take the tourism package.")
    else:
        st.info("Prediction: Customer is unlikely to take the tourism package.")
