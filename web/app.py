from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# === 1) โหลดโมเดลครั้งเดียวตอนเปิดเซิร์ฟเวอร์ ===
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "stroke_model.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
THRESHOLD = bundle["threshold"]
FEATURES = bundle["features"]
print(f"โหลดโมเดลแล้ว | threshold = {THRESHOLD} | sklearn {bundle['sklearn_version']}")

# ค่าที่ยอมรับของแต่ละช่อง (ต้องตรงกับข้อมูลตอนเทรนทุกตัวอักษร)
CHOICES = {
    "gender": ["Male", "Female"],
    "ever_married": ["Yes", "No"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["never smoked", "formerly smoked", "smokes", "Unknown"],
}
RANGES = {"age": (0, 120), "avg_glucose_level": (40, 400), "bmi": (10, 70)}

app = Flask(__name__)
app.json.ensure_ascii = False   # ให้ภาษาไทยใน JSON อ่านออก


def validate(data):
    """ตรวจและแปลงข้อมูลจากหน้าเว็บ คืนค่า (row, error)"""
    row = {}
    for col, allowed in CHOICES.items():
        if col not in FEATURES:          # ข้ามช่องที่โมเดลไม่ได้ใช้
            continue
        value = data.get(col)
        if value not in allowed:
            return None, f"ค่า {col} ไม่ถูกต้อง: {value}"
        row[col] = value

    for col in ["hypertension", "heart_disease"]:
        if data.get(col) not in (0, 1, "0", "1"):
            return None, f"ค่า {col} ต้องเป็น 0 หรือ 1"
        row[col] = int(data[col])

    for col, (low, high) in RANGES.items():
        value = data.get(col)
        if col == "bmi" and value in (None, ""):
            row[col] = np.nan            # ไม่ทราบ BMI -> ให้ Pipeline เติม median เอง
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None, f"ค่า {col} ต้องเป็นตัวเลข"
        if not low <= value <= high:
            return None, f"ค่า {col} ต้องอยู่ระหว่าง {low}-{high}"
        row[col] = value
    return row, None


# === 2) หน้าเว็บ ===
@app.route("/")
def home():
    return render_template("index.html")


# === 3) API สำหรับทำนาย: รับ JSON -> ส่ง JSON กลับ ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row, error = validate(data)
    if error:
        return jsonify({"error": error}), 400

    X = pd.DataFrame([row], columns=FEATURES)          # เรียงคอลัมน์ให้ตรงกับตอนเทรน
    score = float(model.predict_proba(X)[0, 1])
    is_risk = score >= THRESHOLD

    return jsonify({
        "score": round(score, 4),
        "threshold": THRESHOLD,
        "prediction": int(is_risk),
        "label": "มีความเสี่ยง" if is_risk else "ความเสี่ยงต่ำ",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
