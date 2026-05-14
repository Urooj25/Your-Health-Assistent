# 🩺 DiabetesCare AI

> An AI-powered web application that predicts diabetes risk from lab report values — built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-Educational-green?style=flat-square)

---

## 📌 Overview

**DiabetesCare AI** is a machine learning web app that analyzes health indicators and predicts whether a patient is at risk of diabetes. Users can either enter their lab values manually or paste their lab report text directly — the AI automatically extracts the values and runs the prediction.

> ⚕️ **Disclaimer:** This tool is for **educational purposes only**. It is not a substitute for professional medical advice or diagnosis. Always consult a qualified doctor.

---

## ✨ Features

- 🔢 **Manual Entry Mode** — Enter each lab value individually with helpful tooltips
- 📋 **Paste Report Mode** — Paste raw lab report text; AI extracts values automatically using NLP
- 🤖 **Instant Prediction** — Logistic Regression model trained on the PIMA Diabetes Dataset
- 📊 **Confidence Score** — Shows model confidence percentage with every prediction
- 💊 **Personalized Advice** — Displays diet, exercise, and medical precautions based on result
- 🎨 **Dark UI Design** — Clean, modern dark-themed interface built with custom CSS

---

## 🖥️ Demo

Try the live app here:
**🔗 [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)**

---

## 📁 Project Structure

```
diabetes-care-ai/
│
├── app.py                  # Main Streamlit application
├── diabetes_model.pkl      # Trained Logistic Regression model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/diabetes-care-ai.git
cd diabetes-care-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
joblib
numpy
scikit-learn
```

---

## 🧠 How It Works

1. User inputs 7 health values (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age)
2. Values are passed to a pre-trained **Logistic Regression** model
3. Model returns a binary prediction: **Diabetic** or **Non-Diabetic**
4. App displays the result with confidence score and actionable health advice

### Input Features Used

| Feature | Description | Normal Range |
|--------|-------------|--------------|
| Glucose | Fasting blood sugar level | 70–99 mg/dL |
| Blood Pressure | Diastolic BP | Below 80 mmHg |
| Skin Thickness | Triceps skinfold | 10–40 mm |
| Insulin | 2-hour serum insulin | 2–25 mU/L |
| BMI | Body Mass Index | 18.5–24.9 |
| Diabetes Pedigree | Genetic risk score | 0.078–2.42 |
| Age | Patient age in years | — |

---

## 📋 Sample Test Input (Paste Mode)

**High Risk Example:**
```
Patient: Ahmed, Age: 45 years
Fasting Blood Sugar (FBS): 148 mg/dL
Blood Pressure: 130/85 mmHg
BMI: 31.2 kg/m2
Insulin: 120 mU/L
Skin Thickness: 32 mm
Diabetes Pedigree: 0.62
```

**Low Risk Example:**
```
Patient: Sara, Age: 28 years
Fasting Blood Sugar (FBS): 88 mg/dL
Blood Pressure: 110/72 mmHg
BMI: 22.4 kg/m2
Insulin: 40 mU/L
Skin Thickness: 18 mm
Diabetes Pedigree: 0.15
```

---

## 🚀 Deployment (Streamlit Cloud)

1. Push all files to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** and select your repository
4. Set **main file path** to `app.py`
5. Click **"Deploy"** — your app will be live in 2–3 minutes!

---

## 📊 Model Information

| Detail | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| Dataset | PIMA Indians Diabetes Dataset |
| Features | 8 (Pregnancies set to 0 by default) |
| Task | Binary Classification |

---

## 👨‍💻 Author

Built with ❤️ for educational purposes.

---

## 📄 License

This project is intended for **educational use only**. Not for clinical or diagnostic purposes.
