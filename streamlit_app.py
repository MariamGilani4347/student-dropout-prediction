import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Dropout Early-Warning System", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("dropout_model.pkl")


@st.cache_data
def load_columns():
    with open("columns.json") as f:
        return json.load(f)


model = load_model()
COLUMNS = load_columns()

st.title("🎓 Student Dropout Early-Warning System")
st.write("Enter student information to get the dropout probability and risk level.")

with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17.0, 30.0, 21.0, 0.1)
        family_income = st.number_input("Family Income", min_value=0.0, value=35000.0, step=1000.0)
        internet = st.radio("Internet Access", ["Yes", "No"], index=0, horizontal=True)
        study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 4.0, 0.1)
        attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 80.0, 0.1)
        assign_delay = st.slider("Assignment Delay (Days)", 0, 15, 1)
        travel_time = st.slider("Travel Time (Minutes)", 0, 120, 30)
        part_time = st.radio("Part-Time Job", ["Yes", "No"], index=1, horizontal=True)
        scholarship = st.radio("Scholarship", ["Yes", "No"], index=1, horizontal=True)

    with col2:
        stress = st.slider("Stress Index", 0.0, 10.0, 5.0, 0.1)
        gpa = st.slider("GPA", 0.0, 4.0, 2.5, 0.01)
        sem_gpa = st.slider("Semester GPA", 0.0, 4.0, 2.5, 0.01)
        cgpa = st.slider("CGPA", 0.0, 4.0, 2.5, 0.01)
        semester = st.selectbox("Semester (Year)", [1, 2, 3, 4])
        gender = st.radio("Gender", ["Male", "Female"], index=0, horizontal=True)
        department = st.selectbox("Department", ["Arts", "Business", "CS", "Engineering", "Science"], index=2)
        parental_edu = st.selectbox("Parental Education", ["Bachelor", "High School", "Master", "PhD"])

    submitted = st.form_submit_button("Predict Dropout Risk")

if submitted:
    row = pd.DataFrame(0.0, index=[0], columns=COLUMNS)

    row["Age"] = age
    row["Family_Income"] = family_income
    row["Internet_Access"] = 1 if internet == "Yes" else 0
    row["Study_Hours_per_Day"] = study_hours
    row["Attendance_Rate"] = attendance
    row["Assignment_Delay_Days"] = assign_delay
    row["Travel_Time_Minutes"] = travel_time
    row["Part_Time_Job"] = 1 if part_time == "Yes" else 0
    row["Scholarship"] = 1 if scholarship == "Yes" else 0
    row["Stress_Index"] = stress
    row["GPA"] = gpa
    row["Semester_GPA"] = sem_gpa
    row["CGPA"] = cgpa
    row["Semester"] = int(semester)

    if gender == "Male":
        row["Gender_Male"] = 1
    if department != "Arts":
        row["Department_" + department] = 1
    if parental_edu != "Bachelor":
        row["Parental_Education_" + parental_edu] = 1

    prob = float(model.predict_proba(row)[0, 1])

    st.subheader("Result")
    st.metric("Dropout Probability", f"{prob * 100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob < 0.30:
        st.success("🟢 Risk Level: Low Risk")
    elif prob < 0.60:
        st.warning("🟡 Risk Level: Medium Risk")
    else:
        st.error("🔴 Risk Level: High Risk")
