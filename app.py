import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

rf_model = joblib.load('hr_analytics_model.h5')

scaler = StandardScaler()

st.title("Selamat Datang di Model Prediksi Job Satisfaction")

st.header("Masukkan Data")

age = st.number_input("Age")
attrition = st.selectbox("Attrition", ["Yes", "No"])
business_travel = st.selectbox("Business Travel", ["Travel Rarely", "Travel Frequently", "Non-Travel"])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
distance_from_home = st.number_input("Distance From Home")
education = st.number_input("Education")
education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
employee_id = st.number_input("Employee ID")
gender = st.selectbox("Gender", ["Male", "Female"])
job_level = st.number_input("Job Level")
job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
monthly_income = st.number_input("Monthly Income")
num_companies = st.number_input("Num Companies")
percent_salary_hike = st.number_input("Percent Salary Hike")
stock_option_level = st.number_input("Stock Option Level")
total_working_years = st.number_input("Total Working Years")
training_times_last_year = st.number_input("Training Times Last Year")
years_at_company = st.number_input("Years At Company")
years_since_last_promotion = st.number_input("Years Since Last Promotion")
years_with_curr_manager = st.number_input("Years With Curr Manager")
environment_satisfaction = st.number_input("Environment Satisfaction")
work_life_balance = st.number_input("Work Life Balance")
job_involvement = st.number_input("Job Involvement")
performance_rating = st.number_input("Performance Rating")

btn = st.button("predict")

if btn:
    # Map categorical inputs to one-hot encoded format
    input_data = pd.DataFrame({
        'Age': [age],
        'DistanceFromHome': [distance_from_home],
        'Education': [education],
        'JobLevel': [job_level],
        'MonthlyIncome': [monthly_income],
        'NumCompaniesWorked': [num_companies],
        'PercentSalaryHike': [percent_salary_hike],
        'StockOptionLevel': [stock_option_level],
        'TotalWorkingYears': [total_working_years],
        'TrainingTimesLastYear': [training_times_last_year],
        'YearsAtCompany': [years_at_company],
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'YearsWithCurrManager': [years_with_curr_manager],
        'EnvironmentSatisfaction': [environment_satisfaction],
        'WorkLifeBalance': [work_life_balance],
        'JobInvolvement': [job_involvement],
        'PerformanceRating': [performance_rating],
        'Attrition_Yes': [1 if attrition == 'Yes' else 0],
        'BusinessTravel_Travel_Frequently': [1 if business_travel == 'Travel Frequently' else 0],
        'BusinessTravel_Travel_Rarely': [1 if business_travel == 'Travel Rarely' else 0],
        'Department_Research & Development': [1 if department == 'Research & Development' else 0],
        'Department_Sales': [1 if department == 'Sales' else 0],
        'EducationField_Life Sciences': [1 if education_field == 'Life Sciences' else 0],
        'EducationField_Marketing': [1 if education_field == 'Marketing' else 0],
        'EducationField_Medical': [1 if education_field == 'Medical' else 0],
        'EducationField_Other': [1 if education_field == 'Other' else 0],
        'EducationField_Technical Degree': [1 if education_field == 'Technical Degree' else 0],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'JobRole_Human Resources': [1 if job_role == 'Human Resources' else 0],
        'JobRole_Laboratory Technician': [1 if job_role == 'Laboratory Technician' else 0],
        'JobRole_Manager': [1 if job_role == 'Manager' else 0],
        'JobRole_Manufacturing Director': [1 if job_role == 'Manufacturing Director' else 0],
        'JobRole_Research Director': [1 if job_role == 'Research Director' else 0],
        'JobRole_Research Scientist': [1 if job_role == 'Research Scientist' else 0],
        'JobRole_Sales Executive': [1 if job_role == 'Sales Executive' else 0],
        'JobRole_Sales Representative': [1 if job_role == 'Sales Representative' else 0],
        'MaritalStatus_Married': [1 if marital_status == 'Married' else 0],
        'MaritalStatus_Single': [1 if marital_status == 'Single' else 0]
    })

    # Standardize numerical features
    numerical_features = input_data.columns
    processed_input_scaled = scaler.fit_transform(input_data[numerical_features])

    # Predict using the trained model
    prediction = rf_model.predict(processed_input_scaled)

    st.subheader(f"Predicted Job Satisfaction: {prediction[0]}")