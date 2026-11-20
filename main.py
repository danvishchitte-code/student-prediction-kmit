# from pathlib import Path
# import pickle

# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel

# # Load the saved model pipeline from the same folder as main.py
# MODEL_PATH = Path(__file__).with_name("student_dropout_model.pkl")

# with MODEL_PATH.open("rb") as file:
#     model = pickle.load(file)

# # Exact input columns and their order, taken from the saved pipeline
# FEATURES = list(model.feature_names_in_)

# app = FastAPI(title="Student Dropout Prediction API")


# class StudentData(BaseModel):
#     marital_status: float
#     application_mode: float
#     application_order: float
#     course: float
#     daytime_evening_attendance: float
#     previous_qualification: float
#     previous_qualification_grade: float
#     nacionality: float
#     mothers_qualification: float
#     fathers_qualification: float
#     mothers_occupation: float
#     fathers_occupation: float
#     admission_grade: float
#     displaced: float
#     educational_special_needs: float
#     debtor: float
#     tuition_fees_up_to_date: float
#     gender: float
#     scholarship_holder: float
#     age_at_enrollment: float
#     international: float
#     curricular_units_1st_sem_credited: float
#     curricular_units_1st_sem_enrolled: float
#     curricular_units_1st_sem_evaluations: float
#     curricular_units_1st_sem_approved: float
#     curricular_units_1st_sem_grade: float
#     curricular_units_1st_sem_without_evaluations: float
#     unemployment_rate: float
#     inflation_rate: float
#     gdp: float


# # Maps the API's JSON field names to the exact column names the model was trained on
# ALIASES = {
#     "marital_status": "Marital status",
#     "application_mode": "Application mode",
#     "application_order": "Application order",
#     "course": "Course",
#     "daytime_evening_attendance": "Daytime/evening attendance",
#     "previous_qualification": "Previous qualification",
#     "previous_qualification_grade": "Previous qualification (grade)",
#     "nacionality": "Nacionality",
#     "mothers_qualification": "Mother's qualification",
#     "fathers_qualification": "Father's qualification",
#     "mothers_occupation": "Mother's occupation",
#     "fathers_occupation": "Father's occupation",
#     "admission_grade": "Admission grade",
#     "displaced": "Displaced",
#     "educational_special_needs": "Educational special needs",
#     "debtor": "Debtor",
#     "tuition_fees_up_to_date": "Tuition fees up to date",
#     "gender": "Gender",
#     "scholarship_holder": "Scholarship holder",
#     "age_at_enrollment": "Age at enrollment",
#     "international": "International",
#     "curricular_units_1st_sem_credited": "Curricular units 1st sem (credited)",
#     "curricular_units_1st_sem_enrolled": "Curricular units 1st sem (enrolled)",
#     "curricular_units_1st_sem_evaluations": "Curricular units 1st sem (evaluations)",
#     "curricular_units_1st_sem_approved": "Curricular units 1st sem (approved)",
#     "curricular_units_1st_sem_grade": "Curricular units 1st sem (grade)",
#     "curricular_units_1st_sem_without_evaluations":
#         "Curricular units 1st sem (without evaluations)",
#     "unemployment_rate": "Unemployment rate",
#     "inflation_rate": "Inflation rate",
#     "gdp": "GDP",
# }


# @app.get("/")
# def home():
#     return {"message": "Student Dropout Prediction API is running"}


# @app.get("/health")
# def health():
#     return {"status": "healthy"}


# @app.post("/predict")
# def predict(student: StudentData):
#     # Convert API-friendly names to the exact dataset column names
#     values = student.model_dump()
#     row = {ALIASES[key]: value for key, value in values.items()}

#     # Arrange columns in the exact order used during training
#     input_df = pd.DataFrame([row], columns=FEATURES)

#     prediction = int(model.predict(input_df)[0])
#     probability = float(model.predict_proba(input_df)[0, 1])
#     return {
#         "prediction": prediction,
#         "status": "Dropout" if prediction == 1 else "Not Dropout",
#         "dropout_probability": round(probability, 4),
#     }



from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Load the saved model pipeline from the same folder as main.py
MODEL_PATH = Path(__file__).with_name("student_dropout_model.pkl")

with MODEL_PATH.open("rb") as file:
    model = pickle.load(file)

# Exact input columns and their order, taken from the saved pipeline
FEATURES = list(model.feature_names_in_)

app = FastAPI(title="Student Dropout Prediction API")


class StudentData(BaseModel):
    # Pre-filled with realistic sample values so evaluators can test right away
    marital_status: float = Field(default=1.0, description="Marital status code (e.g., 1 = single)")
    application_mode: float = Field(default=1.0, description="Application mode code (e.g., 1 = 1st phase general contingent)")
    application_order: float = Field(default=1.0, description="Application order (0-9)")
    course: float = Field(default=9500.0, description="Course code (e.g., 9500)")
    daytime_evening_attendance: float = Field(default=1.0, description="1 = daytime, 0 = evening")
    previous_qualification: float = Field(default=1.0, description="Previous qualification code (e.g., 1 = secondary education)")
    previous_qualification_grade: float = Field(default=140.0, description="Previous qualification grade (95-190)")
    nacionality: float = Field(default=1.0, description="Nationality code (e.g., 1 = Portuguese)")
    mothers_qualification: float = Field(default=1.0, description="Mother's qualification code")
    fathers_qualification: float = Field(default=1.0, description="Father's qualification code")
    mothers_occupation: float = Field(default=4.0, description="Mother's occupation code")
    fathers_occupation: float = Field(default=7.0, description="Father's occupation code")
    admission_grade: float = Field(default=135.0, description="Admission grade (95-190)")
    displaced: float = Field(default=1.0, description="1 = displaced student, 0 = no")
    educational_special_needs: float = Field(default=0.0, description="1 = yes, 0 = no")
    debtor: float = Field(default=0.0, description="1 = debtor, 0 = no")
    tuition_fees_up_to_date: float = Field(default=1.0, description="1 = up to date, 0 = not up to date")
    gender: float = Field(default=0.0, description="1 = male, 0 = female")
    scholarship_holder: float = Field(default=1.0, description="1 = scholarship holder, 0 = no")
    age_at_enrollment: float = Field(default=18.0, description="Age at enrollment (years)")
    international: float = Field(default=0.0, description="1 = international, 0 = no")
    curricular_units_1st_sem_credited: float = Field(default=0.0, description="1st sem credited units")
    curricular_units_1st_sem_enrolled: float = Field(default=6.0, description="1st sem enrolled units")
    curricular_units_1st_sem_evaluations: float = Field(default=8.0, description="1st sem evaluations")
    curricular_units_1st_sem_approved: float = Field(default=6.0, description="1st sem approved units")
    curricular_units_1st_sem_grade: float = Field(default=14.0, description="1st sem grade (0-20)")
    curricular_units_1st_sem_without_evaluations: float = Field(default=0.0, description="1st sem units without evaluations")
    unemployment_rate: float = Field(default=11.1, description="Unemployment rate (%)")
    inflation_rate: float = Field(default=1.4, description="Inflation rate (%)")
    gdp: float = Field(default=0.32, description="GDP indicator")


# Maps the API's JSON field names to the exact column names the model was trained on
ALIASES = {
    "marital_status": "Marital status",
    "application_mode": "Application mode",
    "application_order": "Application order",
    "course": "Course",
    "daytime_evening_attendance": "Daytime/evening attendance",
    "previous_qualification": "Previous qualification",
    "previous_qualification_grade": "Previous qualification (grade)",
    "nacionality": "Nacionality",
    "mothers_qualification": "Mother's qualification",
    "fathers_qualification": "Father's qualification",
    "mothers_occupation": "Mother's occupation",
    "fathers_occupation": "Father's occupation",
    "admission_grade": "Admission grade",
    "displaced": "Displaced",
    "educational_special_needs": "Educational special needs",
    "debtor": "Debtor",
    "tuition_fees_up_to_date": "Tuition fees up to date",
    "gender": "Gender",
    "scholarship_holder": "Scholarship holder",
    "age_at_enrollment": "Age at enrollment",
    "international": "International",
    "curricular_units_1st_sem_credited": "Curricular units 1st sem (credited)",
    "curricular_units_1st_sem_enrolled": "Curricular units 1st sem (enrolled)",
    "curricular_units_1st_sem_evaluations": "Curricular units 1st sem (evaluations)",
    "curricular_units_1st_sem_approved": "Curricular units 1st sem (approved)",
    "curricular_units_1st_sem_grade": "Curricular units 1st sem (grade)",
    "curricular_units_1st_sem_without_evaluations":
        "Curricular units 1st sem (without evaluations)",
    "unemployment_rate": "Unemployment rate",
    "inflation_rate": "Inflation rate",
    "gdp": "GDP",
}


def run_prediction(student: StudentData) -> dict:
    values = student.model_dump()
    row = {ALIASES[key]: value for key, value in values.items()}
    input_df = pd.DataFrame([row], columns=FEATURES)

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    return {
        "prediction": prediction,
        "status": "Dropout" if prediction == 1 else "Not Dropout",
        "dropout_probability": round(probability, 4),
    }


@app.get("/", response_class=HTMLResponse)
def home():
    defaults = StudentData().model_dump()
    fields_html = []
    for field_name, default_val in defaults.items():
        label_text = ALIASES[field_name]
        desc_text = StudentData.model_fields[field_name].description or ""
        fields_html.append(f"""
            <div class="field">
                <label for="{field_name}">{label_text}</label>
                <input type="number" step="any" id="{field_name}" name="{field_name}" value="{default_val}" required>
                <small>{desc_text}</small>
            </div>
        """)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Student Dropout Prediction</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 820px; margin: 32px auto; padding: 0 16px; }}
            .hint {{ background: #f4f6f8; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }}
            .field {{ display: flex; flex-direction: column; }}
            label {{ font-weight: bold; font-size: 0.9em; margin-bottom: 4px; }}
            input {{ padding: 7px; border: 1px solid #ccc; border-radius: 4px; }}
            small {{ color: #666; font-size: 0.78em; margin-top: 2px; }}
            .actions {{ margin-top: 20px; display: flex; gap: 12px; }}
            button {{ padding: 10px 18px; cursor: pointer; font-size: 1em; }}
        </style>
    </head>
    <body>
        <h2>Student Dropout Prediction</h2>
        <div class="hint">
            <b>Evaluator Guide:</b> Sample student values are pre-filled so you can click <b>Predict</b> right away or edit any value to test different cases.
            You can also inspect the JSON API docs at <a href="/docs">/docs</a>.
        </div>
        <form action="/predict-form" method="post">
            <div class="grid">
                {"".join(fields_html)}
            </div>
            <div class="actions">
                <button type="submit">Predict</button>
                <button type="reset">Reset to Default Sample</button>
            </div>
        </form>
    </body>
    </html>
    """


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    marital_status: float = Form(1.0),
    application_mode: float = Form(1.0),
    application_order: float = Form(1.0),
    course: float = Form(9500.0),
    daytime_evening_attendance: float = Form(1.0),
    previous_qualification: float = Form(1.0),
    previous_qualification_grade: float = Form(140.0),
    nacionality: float = Form(1.0),
    mothers_qualification: float = Form(1.0),
    fathers_qualification: float = Form(1.0),
    mothers_occupation: float = Form(4.0),
    fathers_occupation: float = Form(7.0),
    admission_grade: float = Form(135.0),
    displaced: float = Form(1.0),
    educational_special_needs: float = Form(0.0),
    debtor: float = Form(0.0),
    tuition_fees_up_to_date: float = Form(1.0),
    gender: float = Form(0.0),
    scholarship_holder: float = Form(1.0),
    age_at_enrollment: float = Form(18.0),
    international: float = Form(0.0),
    curricular_units_1st_sem_credited: float = Form(0.0),
    curricular_units_1st_sem_enrolled: float = Form(6.0),
    curricular_units_1st_sem_evaluations: float = Form(8.0),
    curricular_units_1st_sem_approved: float = Form(6.0),
    curricular_units_1st_sem_grade: float = Form(14.0),
    curricular_units_1st_sem_without_evaluations: float = Form(0.0),
    unemployment_rate: float = Form(11.1),
    inflation_rate: float = Form(1.4),
    gdp: float = Form(0.32),
):
    student = StudentData(
        marital_status=marital_status,
        application_mode=application_mode,
        application_order=application_order,
        course=course,
        daytime_evening_attendance=daytime_evening_attendance,
        previous_qualification=previous_qualification,
        previous_qualification_grade=previous_qualification_grade,
        nacionality=nacionality,
        mothers_qualification=mothers_qualification,
        fathers_qualification=fathers_qualification,
        mothers_occupation=mothers_occupation,
        fathers_occupation=fathers_occupation,
        admission_grade=admission_grade,
        displaced=displaced,
        educational_special_needs=educational_special_needs,
        debtor=debtor,
        tuition_fees_up_to_date=tuition_fees_up_to_date,
        gender=gender,
        scholarship_holder=scholarship_holder,
        age_at_enrollment=age_at_enrollment,
        international=international,
        curricular_units_1st_sem_credited=curricular_units_1st_sem_credited,
        curricular_units_1st_sem_enrolled=curricular_units_1st_sem_enrolled,
        curricular_units_1st_sem_evaluations=curricular_units_1st_sem_evaluations,
        curricular_units_1st_sem_approved=curricular_units_1st_sem_approved,
        curricular_units_1st_sem_grade=curricular_units_1st_sem_grade,
        curricular_units_1st_sem_without_evaluations=curricular_units_1st_sem_without_evaluations,
        unemployment_rate=unemployment_rate,
        inflation_rate=inflation_rate,
        gdp=gdp,
    )
    res = run_prediction(student)
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>Prediction Result</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px;">
        <h2>Prediction Result</h2>
        <p><b>Status:</b> {res["status"]} (Class: {res["prediction"]})</p>
        <p><b>Dropout Probability:</b> {res["dropout_probability"]}</p>
        <a href="/">Back to Form</a>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(student: StudentData):
    return run_prediction(student)
