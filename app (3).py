import streamlit as st
import joblib
import numpy as np
import re
import os
import json
import math
from datetime import datetime
from io import BytesIO
import plotly.graph_objects as go

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesCare AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CSS Styling — Dark Slate + Gold Theme ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(145deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    min-height: 100vh;
}

/* ── Hero ── */
.hero-wrap {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.9rem;
    background: linear-gradient(135deg, #f5c518, #e8a000, #f5c518);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #b0b8c8;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 1rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(245,197,24,0.12);
    border: 1px solid rgba(245,197,24,0.35);
    color: #f5c518;
    border-radius: 20px;
    padding: 0.25rem 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── Lang Toggle ── */
.lang-bar {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.5rem;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(245,197,24,0.18);
    border-radius: 18px;
    padding: 1.6rem;
    margin-bottom: 1.3rem;
    backdrop-filter: blur(12px);
}
.card-gold {
    background: rgba(245,197,24,0.06);
    border: 1px solid rgba(245,197,24,0.3);
    border-radius: 18px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
}

/* ── Section label ── */
.mode-label {
    color: #f5c518;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.7rem;
}

/* ── Info box ── */
.info-box {
    background: rgba(245,197,24,0.07);
    border: 1px solid rgba(245,197,24,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #c8cdd8;
    font-size: 0.87rem;
    margin-bottom: 1rem;
}

/* ── Results ── */
.result-diabetic {
    background: linear-gradient(135deg, rgba(220,53,69,0.18), rgba(220,53,69,0.06));
    border: 2px solid rgba(220,53,69,0.55);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-safe {
    background: linear-gradient(135deg, rgba(40,200,100,0.15), rgba(40,200,100,0.04));
    border: 2px solid rgba(40,200,100,0.5);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-icon { font-size: 3.5rem; }
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #f0e6c0;
    margin: 0.5rem 0;
}
.result-desc { color: #b8c8d8; font-size: 0.95rem; }
.confidence-gold { color: #f5c518; font-weight: 700; font-size: 1.1rem; }

/* ── Precaution / tip items ── */
.precaution-item {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #f5c518;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #c8cdd8;
    font-size: 0.9rem;
}
.safe-item {
    background: rgba(40,200,100,0.06);
    border-left: 3px solid #28c864;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #c8cdd8;
    font-size: 0.9rem;
}

/* ── History items ── */
.history-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(245,197,24,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    color: #c8cdd8;
    font-size: 0.9rem;
}
.history-safe { border-left: 4px solid #28c864; }
.history-risk { border-left: 4px solid #dc3545; }
.history-time { color: #f5c518; font-size: 0.78rem; font-weight: 600; }

/* ── BMI result colors ── */
.bmi-normal { color: #28c864; font-weight: 700; }
.bmi-over { color: #f5c518; font-weight: 700; }
.bmi-obese { color: #dc3545; font-weight: 700; }
.bmi-under { color: #7fb3ff; font-weight: 700; }

/* ── Buttons ── */
.stButton>button {
    background: linear-gradient(135deg, #c9a227, #f5c518, #c9a227) !important;
    color: #1a1a2e !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: all 0.25s !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(245,197,24,0.35) !important;
}

/* ── Inputs ── */
div[data-testid="stTextArea"] textarea {
    background: rgba(15,20,40,0.9) !important;
    border: 1px solid rgba(245,197,24,0.25) !important;
    color: #e8e8f0 !important;
    -webkit-text-fill-color: #e8e8f0 !important;
    caret-color: #f5c518 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stNumberInput"] input,
input[type="number"] {
    background: rgba(15,20,40,0.9) !important;
    border: 1px solid rgba(245,197,24,0.25) !important;
    color: #e8e8f0 !important;
    -webkit-text-fill-color: #e8e8f0 !important;
    caret-color: #f5c518 !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] input::selection,
div[data-testid="stTextArea"] textarea::selection {
    background: rgba(245,197,24,0.35) !important;
    color: #ffffff !important;
}

/* ── Labels & text ── */
label, .stSelectbox label, .stRadio label {
    color: #c8a84b !important;
    font-weight: 500 !important;
}
.stRadio div { color: #c8cdd8 !important; }
h2, h3 {
    color: #f0e6c0 !important;
    font-family: 'Playfair Display', serif !important;
}

/* ── Divider ── */
hr { border-color: rgba(245,197,24,0.15) !important; }

/* ── Gauge SVG container ── */
.gauge-wrap { display: flex; justify-content: center; margin: 1rem 0; }

/* ── Download button special ── */
.dl-btn {
    display: inline-block;
    background: linear-gradient(135deg, #c9a227, #f5c518);
    color: #1a1a2e !important;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    font-size: 0.9rem;
    text-decoration: none;
    margin-top: 0.5rem;
    cursor: pointer;
}

/* ── Expander ── */
details summary {
    color: #f5c518 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ─── Session State Init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

# ─── Language Strings ─────────────────────────────────────────────────────────
STRINGS = {
    "EN": {
        "title": "🩺 DiabetesCare AI",
        "subtitle": "Enter your lab values manually or paste your report — AI predicts your diabetes risk",
        "badge": "AI-Powered • Educational Tool",
        "disclaimer": "⚕️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Not a substitute for professional medical diagnosis. Consult a qualified doctor for health concerns.",
        "choose_input": "Choose input method:",
        "manual": "🔢 Manual Entry",
        "paste": "📋 Paste Lab Report",
        "enter_values": "🔢 Enter Your Lab Values",
        "fill_info": "Fill in each value from your latest blood test report. Hover over <b>ℹ️</b> for guidance.",
        "predict_btn": "🔍 Predict Diabetes Risk",
        "history_title": "🕒 Previous Checks (This Session)",
        "no_history": "No predictions yet in this session.",
        "bmi_calc": "🧮 BMI Calculator",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "calc_bmi": "Calculate BMI",
        "your_bmi": "Your BMI",
        "download_report": "📥 Download PDF Report",
        "lang_toggle": "اردو",
        "risk_detected": "Diabetes Risk Detected",
        "no_risk": "No Diabetes Detected",
        "consult": "Please consult your doctor as soon as possible and follow the precautions below.",
        "great_news": "Great news! No diabetes risk detected — keep up your healthy habits.",
        "confidence": "Model Confidence",
        "precautions": "📋 Recommended Precautions",
        "stay_healthy": "💛 Tips to Stay Healthy",
        "important_note": "⚠️ **Important:** This is an AI-based prediction, not a medical diagnosis. Please visit a qualified doctor.",
        "checkup_note": "ℹ️ Stay consistent with regular checkups, especially if you have a family history of diabetes.",
        "values_extracted": "✅ Values extracted from your report:",
        "missing_warning": "⚠️ These values were not found. Please enter them manually:",
        "paste_info": "📋 Copy your lab report and paste it below using <b>Ctrl+V</b>. AI will auto-extract your health values.",
        "paste_label": "Paste your lab report, doctor notes, or any text with health values:",
        "complete_values": "⚠️ Please complete all required values before running the prediction.",
        "glucose_error": "Glucose cannot be 0. Please enter a valid blood sugar value.",
        "bmi_error": "BMI cannot be 0. Please enter a valid BMI value.",
        "clear_history": "🗑️ Clear History",
        "risk_gauge": "📊 Risk Gauge",
        "value_chart": "📈 Your Values vs Normal Range",
    },
    "UR": {
        "title": "🩺 ذیابیطس کیئر AI",
        "subtitle": "اپنی لیب ویلیوز درج کریں یا رپورٹ پیسٹ کریں — AI آپ کا ذیابیطس خطرہ بتائے گا",
        "badge": "AI سے چلنے والا • تعلیمی ٹول",
        "disclaimer": "⚕️ <strong>اہم:</strong> یہ ٹول صرف تعلیمی مقاصد کے لیے ہے۔ کسی بھی صحت کے مسئلے کے لیے ڈاکٹر سے ملیں۔",
        "choose_input": "ان پٹ طریقہ منتخب کریں:",
        "manual": "🔢 ہاتھ سے درج کریں",
        "paste": "📋 رپورٹ پیسٹ کریں",
        "enter_values": "🔢 اپنی لیب ویلیوز درج کریں",
        "fill_info": "اپنی تازہ ترین خون کی رپورٹ سے ہر قدر درج کریں۔",
        "predict_btn": "🔍 ذیابیطس خطرہ جانیں",
        "history_title": "🕒 پچھلے چیک (اس سیشن میں)",
        "no_history": "اس سیشن میں ابھی تک کوئی پیشگوئی نہیں ہوئی۔",
        "bmi_calc": "🧮 BMI کیلکولیٹر",
        "height": "قد (سینٹی میٹر)",
        "weight": "وزن (کلوگرام)",
        "calc_bmi": "BMI حساب کریں",
        "your_bmi": "آپ کا BMI",
        "download_report": "📥 PDF رپورٹ ڈاؤن لوڈ کریں",
        "lang_toggle": "English",
        "risk_detected": "ذیابیطس کا خطرہ ملا",
        "no_risk": "ذیابیطس نہیں ملی",
        "consult": "براہ کرم جلد از جلد اپنے ڈاکٹر سے ملیں اور نیچے دی گئی احتیاطیں پڑھیں۔",
        "great_news": "خوشخبری! ذیابیطس کا کوئی خطرہ نہیں — صحت مند عادات جاری رکھیں۔",
        "confidence": "ماڈل اعتماد",
        "precautions": "📋 تجویز کردہ احتیاطیں",
        "stay_healthy": "💛 صحت مند رہنے کے نکات",
        "important_note": "⚠️ **اہم:** یہ AI پر مبنی پیشگوئی ہے، طبی تشخیص نہیں۔ ڈاکٹر سے ضرور ملیں۔",
        "checkup_note": "ℹ️ خاندان میں ذیابیطس ہو تو سالانہ چیک اپ ضروری ہے۔",
        "values_extracted": "✅ رپورٹ سے نکالی گئی ویلیوز:",
        "missing_warning": "⚠️ یہ ویلیوز نہیں ملیں، براہ کرم خود درج کریں:",
        "paste_info": "📋 اپنی رپورٹ نیچے پیسٹ کریں۔ AI خود بخود تمام صحت کی قدریں نکال لے گا۔",
        "paste_label": "اپنی لیب رپورٹ، ڈاکٹر کے نوٹس، یا کوئی بھی متن یہاں پیسٹ کریں:",
        "complete_values": "⚠️ پیشگوئی سے پہلے تمام ضروری ویلیوز مکمل کریں۔",
        "glucose_error": "گلوکوز صفر نہیں ہو سکتا۔ درست قدر درج کریں۔",
        "bmi_error": "BMI صفر نہیں ہو سکتا۔ درست قدر درج کریں۔",
        "clear_history": "🗑️ تاریخ صاف کریں",
        "risk_gauge": "📊 خطرے کا پیمانہ",
        "value_chart": "📈 آپ کی ویلیوز بمقابلہ نارمل حد",
    }
}

L = STRINGS[st.session_state.lang]

# ─── NLP Parser ────────────────────────────────────────────────────────────────
def extract_number(text, patterns):
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(',', '.'))
            except:
                pass
    return None

def parse_report_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    extracted = {}

    v = extract_number(text, [
        r'glucose[:\s=]*(\d+\.?\d*)', r'blood\s*sugar[:\s=]*(\d+\.?\d*)',
        r'fasting\s*(?:glucose|sugar|bs)[:\s=]*(\d+\.?\d*)',
        r'FBS[:\s=]*(\d+\.?\d*)', r'RBS[:\s=]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*mg/dl',
    ])
    if v is not None: extracted['Glucose'] = v

    m2 = re.search(r'(\d+)/(\d+)\s*(?:mm\s*Hg|mmHg)?', text, re.IGNORECASE)
    if m2:
        extracted['BloodPressure'] = float(m2.group(2))
    else:
        v = extract_number(text, [
            r'(?:blood\s*pressure|BP)[:\s=]*(\d+)(?:/\d+)?',
            r'(?:diastolic|DBP)[:\s=]*(\d+)',
        ])
        if v is not None: extracted['BloodPressure'] = v

    v = extract_number(text, [
        r'skin\s*(?:thickness|fold)[:\s=]*(\d+\.?\d*)',
        r'triceps[:\s=]*(\d+\.?\d*)', r'skinfold[:\s=]*(\d+\.?\d*)',
    ])
    if v is not None: extracted['SkinThickness'] = v

    v = extract_number(text, [
        r'insulin[:\s=]*(\d+\.?\d*)', r'serum\s*insulin[:\s=]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*(?:mu/l|mU/L|uIU)',
    ])
    if v is not None: extracted['Insulin'] = v

    v = extract_number(text, [
        r'bmi[:\s=]*(\d+\.?\d*)', r'body\s*mass\s*index[:\s=]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*kg/m',
    ])
    if v is not None: extracted['BMI'] = v

    v = extract_number(text, [
        r'(?:diabetes\s*)?pedigree[:\s=]*(\d+\.?\d*)',
        r'DPF[:\s=]*(\d+\.?\d*)', r'family\s*(?:history\s*)?(?:score|factor)[:\s=]*(\d+\.?\d*)',
        r'heredit[a-z]*[:\s=]*(\d+\.?\d*)',
    ])
    if v is not None: extracted['DiabetesPedigreeFunction'] = v

    v = extract_number(text, [
        r'age[:\s=]*(\d+)', r'(\d+)\s*(?:year|yr)s?\s*(?:old)?',
        r'عمر[:\s=]*(\d+)',
    ])
    if v is not None: extracted['Age'] = v

    all_fields = ['Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    missing = [f for f in all_fields if f not in extracted]
    return extracted, missing

# ─── Precautions Content ───────────────────────────────────────────────────────
DIABETIC_PRECAUTIONS = {
    "🥗 Diet": [
        "Eat low glycemic index foods — brown rice, oats, lentils, and legumes",
        "Avoid sugary foods, white bread, and white rice as much as possible",
        "Have small meals every 3–4 hours and avoid prolonged fasting",
        "Drink plenty of water — at least 8–10 glasses per day",
        "Eliminate processed and packaged foods from your daily diet",
    ],
    "🏃 Exercise": [
        "Walk briskly for at least 30 minutes every day",
        "Exercise moderately at least 5 days a week",
        "Check your blood sugar levels after exercising",
        "Yoga and meditation help reduce stress and regulate blood sugar",
    ],
    "💊 Medical": [
        "Take your prescribed medications on time without skipping doses",
        "Get your HbA1c tested every 3 months",
        "Monitor your blood sugar regularly at home",
        "Get annual checkups for your eyes, kidneys, and feet",
        "Keep your blood pressure and cholesterol under control as well",
    ],
    "🚫 Avoid": [
        "Quit smoking and avoid alcohol completely",
        "Learn stress management techniques — stress raises blood sugar",
        "Never stop your medication on your own without consulting a doctor",
        "Do not ignore any cuts or wounds on your feet",
    ]
}

SAFE_TIPS = [
    "✅ No diabetes detected — but maintain a healthy lifestyle consistently",
    "🥦 Eat a balanced diet — vegetables, fruits, and lean protein",
    "🏃 Stay physically active every day",
    "⚖️ Keep your weight within a healthy range",
    "🩸 If you have a family history of diabetes, get your blood sugar checked once a year",
    "💧 Stay well hydrated and avoid sugary drinks",
    "😴 Get 7–8 hours of quality sleep — it helps regulate blood sugar too",
]

FIELD_LABELS = {
    'Glucose':                  'Glucose (mg/dL)',
    'BloodPressure':            'Blood Pressure — Diastolic (mmHg)',
    'SkinThickness':            'Skin Thickness (mm)',
    'Insulin':                  'Insulin (mU/L)',
    'BMI':                      'BMI (kg/m²)',
    'DiabetesPedigreeFunction': 'Diabetes Pedigree Function',
    'Age':                      'Age (years)'
}

FIELD_DEFAULTS = {
    'Glucose': 120, 'BloodPressure': 70, 'SkinThickness': 20,
    'Insulin': 80, 'BMI': 25.0, 'DiabetesPedigreeFunction': 0.3, 'Age': 30
}

# Normal ranges for chart
NORMAL_RANGES = {
    'Glucose':                  (70, 99,   "mg/dL"),
    'BloodPressure':            (60, 80,   "mmHg"),
    'SkinThickness':            (10, 40,   "mm"),
    'Insulin':                  (2,  25,   "mU/L"),
    'BMI':                      (18.5, 24.9, "kg/m²"),
    'DiabetesPedigreeFunction': (0.078, 0.5, ""),
    'Age':                      (0, 120,   "yrs"),
}

# ─── Helper: Gauge SVG ────────────────────────────────────────────────────────
def render_gauge(confidence, is_diabetic):
    pct = confidence / 100.0
    angle = -135 + pct * 270
    rad = math.radians(angle)
    cx, cy, r = 150, 140, 100
    nx = cx + r * math.cos(rad)
    ny = cy + r * math.sin(rad)
    color = "#dc3545" if is_diabetic else "#28c864"
    label = f"{'Risk' if is_diabetic else 'Safe'}: {confidence}%"
    svg = f"""
    <svg viewBox="0 0 300 180" xmlns="http://www.w3.org/2000/svg" width="300" height="180">
      <defs>
        <linearGradient id="gGreen" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#28c864"/>
          <stop offset="50%" style="stop-color:#f5c518"/>
          <stop offset="100%" style="stop-color:#dc3545"/>
        </linearGradient>
      </defs>
      <!-- Track -->
      <path d="M 38 140 A 112 112 0 0 1 262 140" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="18" stroke-linecap="round"/>
      <!-- Colored arc -->
      <path d="M 38 140 A 112 112 0 0 1 262 140" fill="none" stroke="url(#gGreen)" stroke-width="18" stroke-linecap="round" opacity="0.7"/>
      <!-- Needle -->
      <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="8" fill="{color}"/>
      <!-- Labels -->
      <text x="38" y="165" fill="#28c864" font-size="11" font-family="Inter,sans-serif" text-anchor="middle">Safe</text>
      <text x="150" y="50" fill="#f5c518" font-size="11" font-family="Inter,sans-serif" text-anchor="middle">Medium</text>
      <text x="262" y="165" fill="#dc3545" font-size="11" font-family="Inter,sans-serif" text-anchor="middle">Risk</text>
      <!-- Value -->
      <text x="150" y="158" fill="{color}" font-size="22" font-weight="bold" font-family="Inter,sans-serif" text-anchor="middle">{confidence}%</text>
      <text x="150" y="175" fill="#b0b8c8" font-size="11" font-family="Inter,sans-serif" text-anchor="middle">{label}</text>
    </svg>
    """
    return svg

# ─── Helper: Bar Chart (Plotly) ──────────────────────────────────────────────
def render_bar_chart(values_dict):
    decimal_fields = {"BMI", "DiabetesPedigreeFunction"}
    FIELD_ORDER = ["Glucose","BloodPressure","SkinThickness",
                   "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    short_labels = {
        "Glucose": "Glucose\n(mg/dL)",
        "BloodPressure": "Blood\nPressure\n(mmHg)",
        "SkinThickness": "Skin\nThickness\n(mm)",
        "Insulin": "Insulin\n(mU/L)",
        "BMI": "BMI\n(kg/m²)",
        "DiabetesPedigreeFunction": "Diabetes\nPedigree",
        "Age": "Age\n(yrs)"
    }
    labels, user_vals, colors, hover_texts = [], [], [], []
    norm_mins, norm_maxs = [], []

    for key in FIELD_ORDER:
        val = values_dict.get(key, FIELD_DEFAULTS[key])
        norm_min, norm_max, unit = NORMAL_RANGES[key]
        labels.append(short_labels[key])
        user_vals.append(val)
        norm_mins.append(norm_min)
        norm_maxs.append(norm_max)
        if val > norm_max:
            colors.append("#ef4444")
            status = "⬆ Above Normal"
        elif val < norm_min:
            colors.append("#60a5fa")
            status = "⬇ Below Normal"
        else:
            colors.append("#22c55e")
            status = "✓ Normal"
        val_str = f"{val:.2f}" if key in decimal_fields else f"{val:.0f}"
        hover_texts.append(
            f"<b>{key}</b><br>"
            f"Your Value: <b>{val_str} {unit}</b><br>"
            f"Normal Range: {norm_min}–{norm_max} {unit}<br>"
            f"Status: {status}"
        )

    fig = go.Figure()

    # Normal range band (as error bars visual — use shape boxes)
    for i, key in enumerate(FIELD_ORDER):
        norm_min, norm_max, unit = NORMAL_RANGES[key]
        fig.add_shape(
            type="rect",
            x0=i - 0.4, x1=i + 0.4,
            y0=norm_min, y1=norm_max,
            fillcolor="rgba(245,197,24,0.15)",
            line=dict(color="rgba(245,197,24,0.5)", width=1),
            layer="below"
        )

    # User value bars
    fig.add_trace(go.Bar(
        x=list(range(len(FIELD_ORDER))),
        y=user_vals,
        marker_color=colors,
        marker_opacity=0.9,
        marker_line=dict(width=0),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        width=0.5,
        showlegend=False,
    ))

    # Legend traces (dummy, for legend only)
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="rgba(245,197,24,0.3)",
                         name="Normal Range Band", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#22c55e",
                         name="✓ Normal", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#ef4444",
                         name="⬆ Above Normal", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#60a5fa",
                         name="⬇ Below Normal", showlegend=True))

    fig.update_layout(
        paper_bgcolor="rgba(15,20,40,0.0)",
        plot_bgcolor="rgba(15,20,40,0.85)",
        font=dict(family="Inter, sans-serif", color="#c8cdd8"),
        margin=dict(l=40, r=20, t=20, b=10),
        height=340,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(FIELD_ORDER))),
            ticktext=labels,
            tickfont=dict(size=11, color="#c8a84b"),
            gridcolor="rgba(245,197,24,0.08)",
            linecolor="rgba(245,197,24,0.3)",
        ),
        yaxis=dict(
            gridcolor="rgba(245,197,24,0.1)",
            tickfont=dict(size=10, color="#888"),
            linecolor="rgba(245,197,24,0.2)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.35,
            xanchor="center", x=0.5,
            font=dict(size=11, color="#c8cdd8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.3,
        hoverlabel=dict(
            bgcolor="#1e2a3a",
            bordercolor="#f5c518",
            font=dict(size=12, color="#f0e6c0"),
        ),
    )
    return fig

# ─── Helper: Generate Text Report (for download) ──────────────────────────────
def generate_text_report(values_dict, prediction, confidence, timestamp):
    lines = [
        "=" * 50,
        "       DiabetesCare AI — Health Report",
        "=" * 50,
        f"Date & Time : {timestamp}",
        f"Result      : {'⚠ DIABETES RISK DETECTED' if prediction == 1 else '✓ NO DIABETES DETECTED'}",
        f"Confidence  : {confidence}%",
        "",
        "─" * 50,
        "Your Lab Values:",
        "─" * 50,
    ]
    for k, v in values_dict.items():
        norm_min, norm_max, unit = NORMAL_RANGES[k]
        status = "✓ Normal" if norm_min <= v <= norm_max else ("↑ High" if v > norm_max else "↓ Low")
        lines.append(f"  {FIELD_LABELS.get(k, k):<38} {v:<10.1f} {unit}  [{status}]")
    lines += [
        "",
        "─" * 50,
        "⚕ IMPORTANT DISCLAIMER",
        "─" * 50,
        "This report is generated by an AI educational tool.",
        "It is NOT a medical diagnosis. Please consult a",
        "qualified doctor for any health concerns.",
        "",
        "DiabetesCare AI • PIMA Diabetes Dataset",
        "=" * 50,
    ]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# ─── APP HEADER ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Language toggle
lang_col1, lang_col2 = st.columns([5, 1])
with lang_col2:
    if st.button(L["lang_toggle"], key="lang_btn"):
        st.session_state.lang = "UR" if st.session_state.lang == "EN" else "EN"
        st.rerun()

L = STRINGS[st.session_state.lang]  # refresh after possible toggle

st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-title">{L['title']}</div>
  <div class="hero-sub">{L['subtitle']}</div>
  <div class="hero-badge">{L['badge']}</div>
</div>
""", unsafe_allow_html=True)

# ─── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(f'<div class="info-box">⚕️ {L["disclaimer"]}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ─── BMI CALCULATOR (collapsible) ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander(L["bmi_calc"]):
    st.markdown('<div class="card-gold">', unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    height_cm = bc1.number_input(L["height"], min_value=50, max_value=250, value=170)
    weight_kg = bc2.number_input(L["weight"], min_value=10, max_value=300, value=70)
    if st.button(L["calc_bmi"], key="bmi_calc_btn"):
        if height_cm > 0:
            bmi_val = weight_kg / ((height_cm / 100) ** 2)
            if bmi_val < 18.5:
                cls, cat = "bmi-under", "Underweight (کم وزن)"
            elif bmi_val < 25:
                cls, cat = "bmi-normal", "Normal ✓"
            elif bmi_val < 30:
                cls, cat = "bmi-over", "Overweight (زیادہ وزن)"
            else:
                cls, cat = "bmi-obese", "Obese (موٹاپا) ⚠"
            st.markdown(
                f'<div style="text-align:center;margin-top:0.5rem;">'
                f'<span style="color:#c8a84b;font-size:0.9rem;">{L["your_bmi"]}: </span>'
                f'<span class="{cls}" style="font-size:2rem;">{bmi_val:.1f}</span>'
                f'<span style="color:#888;font-size:0.9rem;margin-left:0.5rem;">— {cat}</span></div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# ─── INPUT MODE ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
input_mode = st.radio(
    L["choose_input"],
    [L["manual"], L["paste"]],
    horizontal=True
)

extracted_vals = {}
manual_vals = {}
final_values = None
current_vals_dict = {}

# ─── Manual Mode ───────────────────────────────────────────────────────────────
if L["manual"] in input_mode:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="mode-label">{L["enter_values"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">{L["fill_info"]}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    manual_vals['Glucose'] = col1.number_input("🩸 Glucose (mg/dL)", 0, 300, 120, help="Fasting blood glucose level. Normal: 70–99 mg/dL")
    manual_vals['BloodPressure'] = col2.number_input("💓 Blood Pressure — Diastolic (mmHg)", 0, 150, 70, help="Lower number in your BP reading. Normal: below 80 mmHg")
    manual_vals['SkinThickness'] = col1.number_input("📏 Skin Thickness (mm)", 0, 100, 20, help="Triceps skinfold thickness. Normal: 10–40 mm")
    manual_vals['Insulin'] = col2.number_input("💉 Insulin (mU/L)", 0, 900, 80, help="2-hour serum insulin. Normal fasting: 2–25 mU/L")
    manual_vals['BMI'] = col1.number_input("⚖️ BMI (kg/m²)", 0.0, 70.0, 25.0, step=0.1, format="%.1f", help="Healthy range: 18.5–24.9. Use BMI Calculator above!")
    manual_vals['DiabetesPedigreeFunction'] = col2.number_input("🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.3, step=0.01, format="%.3f", help="Genetic diabetes likelihood score. Typical: 0.078–2.42")
    manual_vals['Age'] = col1.number_input("🎂 Age (years)", 1, 120, 30, help="Your current age")
    st.markdown('</div>', unsafe_allow_html=True)

    current_vals_dict = manual_vals.copy()
    final_values = [
        0,
        manual_vals['Glucose'], manual_vals['BloodPressure'],
        manual_vals['SkinThickness'], manual_vals['Insulin'],
        manual_vals['BMI'], manual_vals['DiabetesPedigreeFunction'],
        manual_vals['Age']
    ]

# ─── Paste / NLP Mode ──────────────────────────────────────────────────────────
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="mode-label">📄 Paste Your Lab Report</div>', unsafe_allow_html=True)
    example = """Example:\nPatient: Aisha, Age: 34 years\nFasting Blood Sugar (FBS): 145 mg/dL\nBlood Pressure: 120/80 mmHg\nBMI: 27.5 kg/m2\nInsulin: 94 mU/L\nSkin Thickness: 29 mm\nDiabetes Pedigree: 0.35"""
    st.markdown(f'<div class="info-box">{L["paste_info"]}</div>', unsafe_allow_html=True)
    report_text = st.text_area(L["paste_label"], height=220, placeholder=example)
    st.markdown('</div>', unsafe_allow_html=True)

    if report_text.strip():
        extracted_vals, missing = parse_report_text(report_text)
        if extracted_vals:
            st.markdown(f"**{L['values_extracted']}**")
            cols = st.columns(2)
            for i, (k, v) in enumerate(extracted_vals.items()):
                cols[i % 2].success(f"**{FIELD_LABELS.get(k, k)}:** {v}")
        if missing:
            st.warning(f"{L['missing_warning']} **{', '.join([FIELD_LABELS.get(f, f) for f in missing])}**")
            col1, col2 = st.columns(2)
            for i, field in enumerate(missing):
                col = col1 if i % 2 == 0 else col2
                if field in ['BMI', 'DiabetesPedigreeFunction']:
                    extracted_vals[field] = col.number_input(FIELD_LABELS[field], value=FIELD_DEFAULTS[field], step=0.1, format="%.2f")
                else:
                    extracted_vals[field] = col.number_input(FIELD_LABELS[field], value=int(FIELD_DEFAULTS[field]), step=1)
        if len(extracted_vals) >= 7:
            current_vals_dict = extracted_vals.copy()
            final_values = [
                0,
                extracted_vals['Glucose'], extracted_vals['BloodPressure'],
                extracted_vals['SkinThickness'], extracted_vals['Insulin'],
                extracted_vals['BMI'], extracted_vals['DiabetesPedigreeFunction'],
                extracted_vals['Age']
            ]

# ═══════════════════════════════════════════════════════════════════════════════
# ─── PREDICT BUTTON ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

if st.button(L["predict_btn"]):
    if final_values is None or len(final_values) < 8:
        st.error(L["complete_values"])
    else:
        errors = []
        if final_values[1] == 0: errors.append(L["glucose_error"])
        if final_values[5] == 0: errors.append(L["bmi_error"])

        if errors:
            for e in errors: st.warning(f"⚠️ {e}")
        else:
            arr = np.array([final_values])
            prediction = model.predict(arr)[0]
            proba = model.predict_proba(arr)[0]
            confidence = round(proba[prediction] * 100, 1)
            risk_pct = round(proba[1] * 100, 1)  # always diabetes risk probability
            timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
            # Ensure chart dict has all 7 fields in correct order
            FIELD_ORDER = ['Glucose','BloodPressure','SkinThickness',
                           'Insulin','BMI','DiabetesPedigreeFunction','Age']
            chart_vals = {k: float(current_vals_dict.get(k, FIELD_DEFAULTS[k])) for k in FIELD_ORDER}

            # Save to history
            st.session_state.history.append({
                "time": timestamp,
                "prediction": int(prediction),
                "confidence": confidence,
                "values": current_vals_dict.copy()
            })

            st.markdown("---")

            # ── Result Card ──
            if prediction == 1:
                st.markdown(f"""
                <div class="result-diabetic">
                    <div class="result-icon">🔴</div>
                    <div class="result-title">{L['risk_detected']}</div>
                    <div class="result-desc">{L['confidence']}: <span class="confidence-gold">{confidence}%</span></div>
                    <div class="result-desc" style="margin-top:0.5rem;">{L['consult']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div class="result-icon">🟢</div>
                    <div class="result-title">{L['no_risk']}</div>
                    <div class="result-desc">{L['confidence']}: <span class="confidence-gold">{confidence}%</span></div>
                    <div class="result-desc" style="margin-top:0.5rem;">{L['great_news']}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── 📊 Risk Gauge ──
            st.markdown(f"### {L['risk_gauge']}")
            gauge_svg = render_gauge(risk_pct, prediction == 1)
            st.markdown(f'<div class="gauge-wrap">{gauge_svg}</div>', unsafe_allow_html=True)

            # ── 📈 Value vs Normal Range Chart ──
            if current_vals_dict:
                st.markdown(f"### {L['value_chart']}")
                chart_fig = render_bar_chart(chart_vals)
                st.plotly_chart(chart_fig, use_container_width=True, config={"displayModeBar": False})

            # ── Precautions / Tips ──
            if prediction == 1:
                st.markdown(f"### {L['precautions']}")
                for category, items in DIABETIC_PRECAUTIONS.items():
                    with st.expander(f"{category}", expanded=True):
                        for item in items:
                            st.markdown(f'<div class="precaution-item">• {item}</div>', unsafe_allow_html=True)
                st.error(L["important_note"])
            else:
                st.markdown(f"### {L['stay_healthy']}")
                for tip in SAFE_TIPS:
                    st.markdown(f'<div class="safe-item">{tip}</div>', unsafe_allow_html=True)
                st.info(L["checkup_note"])

            # ── 📥 PDF/Text Report Download ──
            st.markdown(f"### {L['download_report']}")
            report_text_content = generate_text_report(
                current_vals_dict, prediction, confidence, timestamp
            )
            st.download_button(
                label="📥 Download My Report (.txt)",
                data=report_text_content.encode("utf-8"),
                file_name=f"DiabetesCare_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
            )

# ═══════════════════════════════════════════════════════════════════════════════
# ─── SESSION HISTORY ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"### {L['history_title']}")

if not st.session_state.history:
    st.markdown(f'<div class="info-box" style="text-align:center;">{L["no_history"]}</div>', unsafe_allow_html=True)
else:
    hcol1, hcol2 = st.columns([4, 1])
    with hcol2:
        if st.button(L["clear_history"], key="clear_hist"):
            st.session_state.history = []
            st.rerun()
    for item in reversed(st.session_state.history):
        cls = "history-risk" if item["prediction"] == 1 else "history-safe"
        icon = "🔴" if item["prediction"] == 1 else "🟢"
        result_label = L["risk_detected"] if item["prediction"] == 1 else L["no_risk"]
        # Show key values summary
        vals_summary = ""
        for k, v in item["values"].items():
            short = {'Glucose': 'Glu', 'BloodPressure': 'BP', 'BMI': 'BMI',
                     'Insulin': 'Ins', 'Age': 'Age', 'SkinThickness': 'Skin',
                     'DiabetesPedigreeFunction': 'DPF'}
            vals_summary += f"<span style='color:#888;font-size:0.8rem;margin-right:0.8rem;'>{short.get(k,k)}: <b style='color:#c8cdd8;'>{v:.0f}</b></span>"
        st.markdown(f"""
        <div class="history-item {cls}">
            <div class="history-time">🕒 {item['time']}</div>
            <div style="margin:0.3rem 0;">{icon} <strong style="color:#f0e6c0;">{result_label}</strong> &mdash; <span class="confidence-gold">{item['confidence']}% confidence</span></div>
            <div style="margin-top:0.3rem;">{vals_summary}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#5a5a7a;font-size:0.8rem;">'
    'DiabetesCare AI &bull; Logistic Regression &bull; PIMA Diabetes Dataset<br>'
    '⚕️ For educational purposes only &mdash; always consult a qualified doctor'
    '</div>',
    unsafe_allow_html=True
)
