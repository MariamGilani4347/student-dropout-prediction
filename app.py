import json
import joblib
import pandas as pd
import gradio as gr

model = joblib.load("dropout_model.pkl")
with open("columns.json") as f:
    COLUMNS = json.load(f)


def predict_dropout(age, family_income, internet, study_hours, attendance,
                    assign_delay, travel_time, part_time, scholarship, stress,
                    gpa, sem_gpa, cgpa, semester, gender, department, parental_edu):

    row = pd.DataFrame(0, index=[0], columns=COLUMNS)

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

    prob = model.predict_proba(row)[0, 1]

    if prob < 0.30:
        risk = "Low Risk"
    elif prob < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return f"{prob * 100:.1f}%", risk


app = gr.Interface(
    fn=predict_dropout,
    inputs=[
        gr.Slider(17, 30, value=21, step=0.1, label="Age"),
        gr.Number(value=35000, label="Family Income"),
        gr.Radio(["Yes", "No"], value="Yes", label="Internet Access"),
        gr.Slider(0, 12, value=4, step=0.1, label="Study Hours per Day"),
        gr.Slider(0, 100, value=80, step=0.1, label="Attendance Rate (%)"),
        gr.Slider(0, 15, value=1, step=1, label="Assignment Delay (Days)"),
        gr.Slider(0, 120, value=30, step=1, label="Travel Time (Minutes)"),
        gr.Radio(["Yes", "No"], value="No", label="Part-Time Job"),
        gr.Radio(["Yes", "No"], value="No", label="Scholarship"),
        gr.Slider(0, 10, value=5, step=0.1, label="Stress Index"),
        gr.Slider(0, 4, value=2.5, step=0.01, label="GPA"),
        gr.Slider(0, 4, value=2.5, step=0.01, label="Semester GPA"),
        gr.Slider(0, 4, value=2.5, step=0.01, label="CGPA"),
        gr.Dropdown(["1", "2", "3", "4"], value="1", label="Semester (Year)"),
        gr.Radio(["Male", "Female"], value="Male", label="Gender"),
        gr.Dropdown(["Arts", "Business", "CS", "Engineering", "Science"], value="CS", label="Department"),
        gr.Dropdown(["Bachelor", "High School", "Master", "PhD"], value="Bachelor", label="Parental Education"),
    ],
    outputs=[
        gr.Textbox(label="Dropout Probability"),
        gr.Textbox(label="Risk Level"),
    ],
    title=" Student Dropout Early-Warning System",
    description="Enter student information to get the dropout probability and risk level.",
)

if __name__ == "__main__":
    app.launch()
