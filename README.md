# 🩺 DiabetesCare AI — Streamlit App

## Kaise chalayein?

### Step 1: Install karein
```bash
pip install -r requirements.txt
```

### Step 2: Run karein
```bash
streamlit run app.py
```

Browser mein khul jaega: `http://localhost:8501`

---

## Features

### 📋 NLP Mode (Report Paste)
- Lab report ya doctor notes seedha paste karein
- AI khud glucose, BP, BMI wagera nikal leta hai
- Jo values na milein woh manually bhar sakte hain

### 🔢 Manual Mode
- Seedha values type karein

### 🔍 Prediction
- **Diabetic ➜** Detailed precautions (Diet, Exercise, Medical, Avoid)
- **Not Diabetic ➜** Prevention tips

---

## NLP — Supported Formats

```
Age: 28 years
Fasting Blood Sugar: 145 mg/dl
Blood Pressure: 120/80 mmHg
BMI: 27.5 kg/m2
Insulin: 94 mU/L
Skin Thickness: 29 mm
Pregnancies: 2
Diabetes Pedigree: 0.35
```

---

## Files
- `app.py` — Main Streamlit app
- `diabetes_model.pkl` — Trained Logistic Regression model
- `requirements.txt` — Dependencies

---

⚕️ *Sirf educational purpose ke liye — final diagnosis ke liye doctor se zaroor milein*
