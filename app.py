import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool

# Load the trained model
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_employee_attrition_model.cbm")
    return model

model = load_model()

st.set_page_config(page_title="Employee Attrition Predictor")
st.title("🚧 Employee Attrition Prediction App")
st.write("Enter employee details to predict if they are likely to leave the company.")

with st.form("attrition_form"):
    age = st.slider("Age", 18, 60, 30)
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    daily_rate = st.number_input("Daily Rate", min_value=0, value=1000)
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    distance_from_home = st.slider("Distance From Home (km)", 1, 30, 5)
    education = st.selectbox("Education", [1,2,3,4,5])
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
    environment_satisfaction = st.selectbox("Environment Satisfaction", [1,2,3,4])
    gender = st.selectbox("Gender", ["Male", "Female"])
    hourly_rate = st.number_input("Hourly Rate", min_value=0, value=50)
    job_involvement = st.selectbox("Job Involvement", [1,2,3,4])
    job_level = st.selectbox("Job Level", [1,2,3,4,5])
    job_role = st.selectbox("Job Role", [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician',
        'Manufacturing Director', 'Healthcare Representative',
        'Manager', 'Sales Representative', 'Research Director',
        'Human Resources'
    ])
    job_satisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
    monthly_rate = st.number_input("Monthly Rate", min_value=0, value=10000)
    num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, value=1)
    over_time = st.selectbox("OverTime", ["Yes", "No"])
    percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=10)
    performance_rating = st.selectbox("Performance Rating", [1,2,3,4])
    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1,2,3,4])
    stock_option_level = st.selectbox("Stock Option Level", [0,1,2,3])
    total_working_years = st.number_input("Total Working Years", min_value=0, value=10)
    training_times_last_year = st.number_input("Training Times Last Year", min_value=0, value=2)
    work_life_balance = st.selectbox("Work Life Balance", [1,2,3,4])
    years_at_company = st.number_input("Years at Company", min_value=0, value=5)
    years_in_current_role = st.number_input("Years in Current Role", min_value=0, value=3)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
    years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, value=2)

    submitted = st.form_submit_button("Predict Attrition")

if submitted:
    input_data = pd.DataFrame({
        "Age": [age],
        "BusinessTravel": [business_travel],
        "DailyRate": [daily_rate],
        "Department": [department],
        "DistanceFromHome": [distance_from_home],
        "Education": [education],
        "EducationField": [education_field],
        "EnvironmentSatisfaction": [environment_satisfaction],
        "Gender": [gender],
        "HourlyRate": [hourly_rate],
        "JobInvolvement": [job_involvement],
        "JobLevel": [job_level],
        "JobRole": [job_role],
        "JobSatisfaction": [job_satisfaction],
        "MaritalStatus": [marital_status],
        "MonthlyIncome": [monthly_income],
        "MonthlyRate": [monthly_rate],
        "NumCompaniesWorked": [num_companies_worked],
        "OverTime": [over_time],
        "PercentSalaryHike": [percent_salary_hike],
        "PerformanceRating": [performance_rating],
        "RelationshipSatisfaction": [relationship_satisfaction],
        "StockOptionLevel": [stock_option_level],
        "TotalWorkingYears": [total_working_years],
        "TrainingTimesLastYear": [training_times_last_year],
        "WorkLifeBalance": [work_life_balance],
        "YearsAtCompany": [years_at_company],
        "YearsInCurrentRole": [years_in_current_role],
        "YearsSinceLastPromotion": [years_since_last_promotion],
        "YearsWithCurrManager": [years_with_curr_manager],
    })

    # Reorder columns to match model feature order exactly
    input_data = input_data[model.feature_names_]

    # Categorical features
    cat_features = [
        "BusinessTravel", "Department", "EducationField", "Gender",
        "JobRole", "MaritalStatus", "OverTime"
    ]

    pool = Pool(data=input_data, cat_features=cat_features)

    prediction = model.predict(pool)[0]
    proba = model.predict_proba(pool)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ The employee is likely to leave the company (Attrition Probability: {proba:.2f})")
    else:
        st.success(f"✅ The employee is likely to stay (Attrition Probability: {proba:.2f})")
