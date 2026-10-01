# Stroke Prediction — Mini Project 344-362 Machine Learning

ระบบคัดกรองความเสี่ยงโรคหลอดเลือดสมอง (Binary Classification) ด้วย scikit-learn
พร้อมเว็บแอป 1 หน้า (Flask + HTML/CSS/JavaScript)

## ข้อมูล
- Kaggle: *Stroke Prediction Dataset* (fedesoriano) → `data/healthcare-dataset-stroke-data.csv`
- 5,110 แถว, target `stroke` (1 = 249 คน / 4.87%)

## โครงสร้าง
```
data/        ชุดข้อมูล
notebooks/   01_eda.ipynb (สำรวจข้อมูล), 02_model.ipynb (preprocessing, เทียบโมเดล, ประเมินผล)
src/         train.py (เทรนโมเดลสุดท้ายซ้ำได้ด้วยคำสั่งเดียว)
models/      stroke_model.joblib (Pipeline + threshold)
web/         app.py (Flask API) + templates/index.html + static/script.js, style.css
```

## วิธีรัน
```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py                  # เทรนและบันทึก models/stroke_model.joblib
python web/app.py                    # เปิด http://localhost:5001
```

## โมเดลสุดท้าย
Logistic Regression (C=0.1, L2, class_weight="balanced"), ฟีเจอร์ 6 ตัว, threshold 0.49
ชุด test: Recall (stroke) 0.82, Precision 0.14, F1 0.24, PR-AUC 0.21, ROC-AUC 0.84

> ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์
