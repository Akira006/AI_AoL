from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# === Load Artifacts ===
MODEL_PATH = r"D:\PythonFlask\artifacts\sleep_model.pkl"
SCALER_PATH = r"D:\PythonFlask\artifacts\scaler.pkl"
ENCODER_PATH = r"D:\PythonFlask\artifacts\label_encoder.pkl"
FEATURE_PATH = r"D:\PythonFlask\artifacts\model_features.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
feature_columns = joblib.load(FEATURE_PATH)

# === Preprocessing Function ===
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    # Handle Blood Pressure "120/80"
    if 'Blood Pressure' in df.columns:
        bp = df["Blood Pressure"].astype(str).str.split("/", expand=True)
        if bp.shape[1] == 2:
            df["Systolic_BP"] = pd.to_numeric(bp[0], errors="coerce")
            df["Diastolic_BP"] = pd.to_numeric(bp[1], errors="coerce")
        df = df.drop(columns=["Blood Pressure"], errors="ignore")

    # Fill empty categorical
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("")

    df_encoded = pd.get_dummies(df, drop_first=True)

    # Reindex to match training features
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df_encoded)

    return df_scaled

# === GET endpoint (biar ga Method Not Allowed) ===
@app.route("/predict", methods=["GET"])
def predict_get():
    return jsonify({
        "message": "Use POST method with JSON to get predictions.",
        "example_payload": {
            "Age": 25,
            "Gender": "Male",
            "Sleep Duration": 7
        }
    })

# === POST endpoint for actual prediction ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        processed = preprocess_input(data)
        pred_encoded = model.predict(processed)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        return jsonify({
            "prediction": pred_label,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        })

# === Run server ===
if __name__ == "__main__":
    app.run(debug=True, port=5000)
