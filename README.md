# 🎓 Student Dropout Prediction

A machine learning project that predicts whether a student is at risk of dropping out, and shows the result as a dropout probability and a risk level (Low / Medium / High). It can be used as an educational early-warning system.

**Live App:** PASTE_YOUR_HUGGING_FACE_LINK_HERE

## 1. Problem Statement
Some students leave their studies before completing them. This project identifies students who may be at risk so that universities can support them early.
- Problem type: Binary classification
- Target variable: Dropout (0 = Not Dropout, 1 = Dropout)

## 2. Dataset
- Source: Kaggle - Student Dropout Prediction Dataset (https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset)
- 10,000 records, 19 columns (18 input features + 1 target)
- Class distribution: 76.46% Not Dropout, 23.54% Dropout (moderately imbalanced)

## 3. Data Preprocessing
- Found 2,011 missing values (Family_Income, Study_Hours_per_Day, Stress_Index, Parental_Education)
- Filled numeric columns with the median and categorical columns with the mode
- Checked duplicates (none found) and removed Student_ID
- Converted Yes/No columns (Internet_Access, Part_Time_Job, Scholarship) to 1/0
- Converted Semester (Year 1-4) to numbers 1-4
- One-hot encoded Gender, Department and Parental_Education
- Final dataset: 10,000 rows, 22 input features + target

## 4. Exploratory Data Analysis
- Dropout students had a much lower average GPA (1.43 vs 2.58)
- Higher stress was linked with dropout (6.32 vs 5.26)
- Attendance was slightly lower for dropouts (79.3% vs 82.5%)
- GPA, Semester GPA and CGPA had the strongest negative correlation with dropout (about -0.45)

## 5. Model
- Algorithm: Logistic Regression (scikit-learn)
- Train/test split: 80/20 (stratified, random_state=42)
- Training rows: 8,000 | Testing rows: 2,000

## 6. Evaluation Results (test set)
- Accuracy: 0.8145
- Precision: 0.6786
- Recall: 0.4034
- F1-score: 0.506
- ROC-AUC: 0.8186
- Confusion matrix: TN = 1439, FP = 90, FN = 281, TP = 190

False negatives (dropout students the model missed) are the most costly errors, because those students would not receive support.

## 7. Prediction App
The Gradio app takes student information as input and returns the dropout probability and a risk level:
- Low Risk: below 30%
- Medium Risk: 30% to 60%
- High Risk: above 60%

## 8. How to Run Locally
1. Clone this repository
2. Install requirements: pip install -r requirements.txt
3. Run: python app.py
4. Open the local link shown in the terminal

## 9. Real-World Use
Universities can use this system as an early-warning tool. Advisors can enter a student's information, see the risk level, and offer counselling, tutoring or financial support before the student drops out.

## 10. Limitations
- The model gives a risk estimate, not a final decision about any student
- Results depend on the quality of the dataset
- Future work: try other models (Random Forest, XGBoost), handle class imbalance, add more features

## Files
- app.py - Gradio web application
- dropout_model.pkl - trained Logistic Regression model
- columns.json - feature columns used during training
- requirements.txt - Python dependencies
- Student_Dropout_Prediction.ipynb - full notebook (cleaning, EDA, training, evaluation)
