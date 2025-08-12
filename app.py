import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==== CONFIGURATION ====
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"  # Replace with your dataset path in repo or cloud
MODEL_PATH = "catboost_employee_attrition_model.cbm"

# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv(WA_Fn-UseC_-HR-Employee-Attrition.csv)
    return df

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(catboost_employee_attrition_model.cbm)
    return model

# Load model & data
model = load_model()
df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Visualization", "Model Prediction", "Model Performance"])

# --- PAGE 1: Data Exploration ---
if page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.write("Dataset Overview:")

    st.write(f"Shape: {df.shape}")
    st.write("Columns and Data Types:")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    st.subheader("Sample Data")
    st.dataframe(df.sample(10))

    st.subheader("Filter Data")
    filter_col = st.selectbox("Filter column", df.columns)

    if pd.api.types.is_numeric_dtype(df[filter_col]):
        min_val = float(df[filter_col].min())
        max_val = float(df[filter_col].max())
        selected_range = st.slider(f"Select range for {filter_col}", min_val, max_val, (min_val, max_val))
        filtered_df = df[(df[filter_col] >= selected_range[0]) & (df[filter_col] <= selected_range[1])]
    else:
        unique_vals = df[filter_col].unique()
        selected_vals = st.multiselect(f"Select values for {filter_col}", unique_vals, default=unique_vals)
        filtered_df = df[df[filter_col].isin(selected_vals)]

    st.write(f"Filtered data shape: {filtered_df.shape}")
    st.dataframe(filtered_df.head(20))

# --- PAGE 2: Visualization ---
elif page == "Visualization":
    st.title("📈 Visualization")

    chart_type = st.selectbox("Choose chart type", ["Histogram", "Boxplot", "Countplot by Category"])

    if chart_type == "Histogram":
        num_col = st.selectbox("Select numeric column", df.select_dtypes(include=np.number).columns)
        bins = st.slider("Number of bins", 5, 100, 20)
        fig, ax = plt.subplots()
        ax.hist(df[num_col], bins=bins, color='skyblue', edgecolor='black')
        ax.set_title(f"Histogram of {num_col}")
        st.pyplot(fig)

    elif chart_type == "Boxplot":
        num_col = st.selectbox("Select numeric column", df.select_dtypes(include=np.number).columns)
        cat_col = st.selectbox("Select categorical column", df.select_dtypes(include=['object', 'category']).columns)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
        ax.set_title(f"Boxplot of {num_col} by {cat_col}")
        st.pyplot(fig)

    elif chart_type == "Countplot by Category":
        cat_col = st.selectbox("Select categorical column", df.select_dtypes(include=['object', 'category']).columns)
        fig, ax = plt.subplots()
        sns.countplot(x=df[cat_col], ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f"Countplot of {cat_col}")
        st.pyplot(fig)

# --- PAGE 3: Model Prediction ---
elif page == "Model Prediction":
    st.title("🤖 Employee Attrition Prediction")
    st.write("Enter employee details to predict if they are likely to leave the company.")

    with st.form("attrition_form"):
        age = st.slider("Age", 18, 60, 30)
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        daily_rate = st.number_input("Daily Rate", min_value=0, value=1000)
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        distance_from_home = st.slider("Distance From Home (km)", 1, 30, 5)
        education = st.selectbox("Education", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        gender = st.selectbox("Gender", ["Male", "Female"])
        hourly_rate = st.number_input("Hourly Rate", min_value=0, value=50)
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_role = st.selectbox("Job Role", [
            'Sales Executive', 'Research Scientist', 'Laboratory Technician',
            'Manufacturing Director', 'Healthcare Representative',
            'Manager', 'Sales Representative', 'Research Director',
            'Human Resources'
        ])
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
        monthly_rate = st.number_input("Monthly Rate", min_value=0, value=10000)
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, value=1)
        over_time = st.selectbox("OverTime", ["Yes", "No"])
        percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=10)
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        total_working_years = st.number_input("Total Working Years", min_value=0, value=10)
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, value=2)
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
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

        input_data = input_data[model.feature_names_]

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

# --- PAGE 4: Model Performance ---
elif page == "Model Performance":
    st.title("📈 Model Performance")

    if "Attrition" not in df.columns:
        st.warning("Dataset does not contain 'Attrition' column for performance evaluation.")
    else:
        # Preprocess dataset for test split & evaluation
        df_copy = df.copy()
        cat_cols = [
            "BusinessTravel", "Department", "EducationField", "Gender",
            "JobRole", "MaritalStatus", "OverTime"
        ]
        for col in cat_cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])

        X = df_copy[model.feature_names_]
        y = df_copy["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        test_pool = Pool(X_test, cat_features=[X_test.columns.get_loc(col) for col in cat_cols if col in X_test.columns])
        y_pred = model.predict(test_pool)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        st.metric("Accuracy", f"{acc:.2%}")

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
