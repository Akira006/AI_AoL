# Sleep Disorder Prediction API

A Flask-based machine learning API to predict sleep disorders based on lifestyle and health data.

The model predicts one of the following classes:
- None (No Sleep Disorder)
- Insomnia
- Sleep Apnea

---

## Project Overview
This project uses a Random Forest Classifier trained on a sleep health dataset.  
The trained model is exposed through a REST API built with Flask.

---

## Tech Stack
- Python
- Flask
- pandas
- NumPy
- scikit-learn
- joblib

---

## Project Structure
PythonFlask/
- ├── app.py
- ├── notebook/
- │ ├── Sleep_health_and_lifestyle_dataset.csv
- │ └── save and load model.ipynb
- ├── artifacts/ # ignored in git
- ├── .gitignore
- └── README.md


---

## Model Training

The model is trained inside the Jupyter Notebook:
- notebook/save and load model.ipynb


### Training Steps (Summary)
- Load dataset
- Clean and preprocess data
- Encode categorical features (One-Hot Encoding)
- Scale features using StandardScaler
- Train Random Forest Classifier
- Evaluate model performance
- Save trained artifacts using `joblib`

Saved artifacts:
- `sleep_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `model_features.pkl`

---

## Setup & Installation

### 1. Create virtual environment (optional)
- python -m venv venv

### 2. Activate virtual environment
- venv\Scripts\activate

### 3. Install dependencies
- pip install flask pandas numpy scikit-learn joblib

## Running the API
- python app.py

## API Usage
### GET /predict

Used to verify that the API is running.

Example response:
{
  "message": "Use POST method with JSON to get predictions.",
  "example_payload": {
    "Age": 25,
    "Gender": "Male",
    "Sleep Duration": 7
  }
}

### POST /predict
Send input data in JSON format to get a prediction.
Example request:
{
  "Age": 29,
  "Gender": "Female",
  "Sleep Duration": 4.8,
  "Quality of Sleep": 3,
  "Physical Activity Level": 20,
  "Stress Level": 8,
  "Blood Pressure": "130/85",
  "Heart Rate": 78,
  "Daily Steps": 3000,
  "Occupation": "Office Worker",
  "BMI Category": "Normal"
}

Example response:
{
  "prediction": "Insomnia",
  "confidence": "72.5%",
  "status": "success"
}

## Notes
This API uses Flask's development server and is not intended for production use.
Predictions are based on statistical patterns and are not a medical diagnosis.
Model artifacts are excluded from GitHub using .gitignore.

