import streamlit as st
import joblib
import numpy as np
import re
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesCare AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2d3d 50%, #0f1923 100%);
    min-height: 100vh;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #e8f4f8;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #7fb3c8;
    text-align: center;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(127,179,200,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}

.result-diabetic {
    background: linear-gradient(135deg, rgba(220,53,69,0.15), rgba(220,53,69,0.05));
    border: 2px solid rgba(220,53,69,0.5);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-safe {
    background: linear-gradient(135deg, rgba(40,167,69,0.15), rgba(40,167,69,0.05));
    border: 2px solid rgba(40,167,69,0.5);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-icon { font-size: 3.5rem; }
.result-title { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #e8f4f8; margin: 0.5rem 0; }
.result-desc { color: #a8c8d8; font-size: 0.95rem; }

.precaution-item {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #7fb3c8;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #c8dde8;
    font-size: 0.9rem;
}
.safe-item {
    background: rgba(40,167,69,0.06);
    border-left: 3px solid #28a745;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #c8dde8;
    font-size: 0.9rem;
}

.mode-label {
    color: #7fb3c8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.badge {
    display: inline-block;
    background: rgba(41,128,185,0.2);
    border: 1px solid rgba(41,128,185,0.4);
    color: #7fb3c8;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
}

.info-box {
    background: rgba(41,128,185,0.08);
    border: 1px solid rgba(41,128,185,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #a8c8d8;
    font-size: 0.88rem;
    margin-bottom: 1rem;
}

/* ── Input fields fix: dark background, light text ── */
div[data-testid="stTextArea"] textarea {
    background: rgba(15, 25, 35, 0.85) !important;
    border: 1px solid rgba(127,179,200,0.3) !important;
    color: #e8f4f8 !important;
    -webkit-text-fill-color: #e8f4f8 !important;
    caret-color: #e8f4f8 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    user-select: text !important;
    -webkit-user-select: text !important;
}
div[data-testid="stNumberInput"] input,
input[type="number"] {
    background: rgba(15, 25, 35, 0.85) !important;
    border: 1px solid rgba(127,179,200,0.3) !important;
    color: #e8f4f8 !important;
    -webkit-text-fill-color: #e8f4f8 !important;
    caret-color: #e8f4f8 !important;
    border-radius: 8px !important;
    user-select: text !important;
    -webkit-user-select: text !important;
}
div[data-testid="stNumberInput"] input::selection,
div[data-testid="stTextArea"] textarea::selection {
    background: rgba(127,179,200,0.4) !important;
    color: #ffffff !important;
}

.stButton>button {
    background: linear-gradient(135deg, #2980b9, #1a5f7a) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(41,128,185,0.4) !important;
}
label, .stSelectbox label, .stRadio label {
    color: #7fb3c8 !important;
    font-weight: 500 !important;
}
.stRadio div { color: #c8dde8 !important; }
h2, h3 { color: #e8f4f8 !important; font-family: 'DM Serif Display', serif !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
    return joblib.load(model_path)

model = load_model()

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

    # Glucose
    v = extract_number(text, [
        r'glucose[:\s=]*(\d+\.?\d*)',
        r'blood\s*sugar[:\s=]*(\d+\.?\d*)',
        r'fasting\s*(?:glucose|sugar|bs)[:\s=]*(\d+\.?\d*)',
        r'FBS[:\s=]*(\d+\.?\d*)',
        r'RBS[:\s=]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*mg/dl',
    ])
    if v is not None: extracted['Glucose'] = v

    # Blood Pressure
    m2 = re.search(r'(\d+)/(\d+)\s*(?:mm\s*Hg|mmHg)?', text, re.IGNORECASE)
    if m2:
        extracted['BloodPressure'] = float(m2.group(2))
    else:
        v = extract_number(text, [
            r'(?:blood\s*pressure|BP)[:\s=]*(\d+)(?:/\d+)?',
            r'(?:diastolic|DBP)[:\s=]*(\d+)',
        ])
        if v is not None: extracted['BloodPressure'] = v

    # Skin Thickness
    v = extract_number(text, [
        r'skin\s*(?:thickness|fold)[:\s=]*(\d+\.?\d*)',
        r'triceps[:\s=]*(\d+\.?\d*)',
        r'skinfold[:\s=]*(\d+\.?\d*)',
    ])
    if v is not None: extracted['SkinThickness'] = v

    # Insulin
    v = extract_number(text, [
        r'insulin[:\s=]*(\d+\.?\d*)',
        r'serum\s*insulin[:\s=]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*(?:mu/l|mU/L|uIU)',
    ])
    if v is not None: extracted['Insulin'] = v

    # BMI
    v = extract_number(text, [
        r'bmi[:\s=]*(\d+\.?\d*)',
        r'body\s*mass\s*index[:\s=]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*kg/m',
    ])
    if v is not None: extracted['BMI'] = v

    # Diabetes Pedigree Function
    v = extract_number(text, [
        r'(?:diabetes\s*)?pedigree[:\s=]*(\d+\.?\d*)',
        r'DPF[:\s=]*(\d+\.?\d*)',
        r'family\s*(?:history\s*)?(?:score|factor)[:\s=]*(\d+\.?\d*)',
        r'heredit[a-z]*[:\s=]*(\d+\.?\d*)',
    ])
    if v is not None: extracted['DiabetesPedigreeFunction'] = v

    # Age
    v = extract_number(text, [
        r'age[:\s=]*(\d+)',
        r'(\d+)\s*(?:year|yr)s?\s*(?:old)?',
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

# ─── App Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🩺 DiabetesCare AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Enter your lab values manually or paste your report — AI will predict your diabetes risk</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ─── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="info-box">⚕️ <strong>Disclaimer:</strong> This tool is for educational purposes only. '
    'It is not a substitute for professional medical diagnosis. Please consult a qualified doctor for any health concerns.</div>',
    unsafe_allow_html=True
)

# ─── Input Mode Selection ──────────────────────────────────────────────────────
input_mode = st.radio(
    "Choose input method:",
    ["🔢 Manual Entry", "📋 Paste Lab Report (Text)"],
    horizontal=True
)

extracted_vals = {}
manual_vals = {}
final_values = None

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

# ─── Manual Mode ───────────────────────────────────────────────────────────────
if "Manual" in input_mode:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="mode-label">🔢 Enter Your Lab Values</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Fill in each value from your latest blood test report. Hover over the <b>ℹ️</b> icon on each field for guidance.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    manual_vals['Glucose'] = col1.number_input(
        "🩸 Glucose (mg/dL)", 0, 300, 120,
        help="Fasting blood glucose level from your blood test report (normal range: 70–99 mg/dL)"
    )
    manual_vals['BloodPressure'] = col2.number_input(
        "💓 Blood Pressure — Diastolic (mmHg)", 0, 150, 70,
        help="The lower number in your BP reading (e.g., 80 in 120/80). Normal: below 80 mmHg"
    )
    manual_vals['SkinThickness'] = col1.number_input(
        "📏 Skin Thickness (mm)", 0, 100, 20,
        help="Triceps skinfold thickness measured by your doctor. Normal range: 10–40 mm"
    )
    manual_vals['Insulin'] = col2.number_input(
        "💉 Insulin (mU/L)", 0, 900, 80,
        help="2-hour serum insulin level from your blood test. Normal fasting range: 2–25 mU/L"
    )
    manual_vals['BMI'] = col1.number_input(
        "⚖️ BMI (kg/m²)", 0.0, 70.0, 25.0, step=0.1, format="%.1f",
        help="Body Mass Index = weight(kg) / height(m)². Healthy range: 18.5–24.9"
    )
    manual_vals['DiabetesPedigreeFunction'] = col2.number_input(
        "🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.3, step=0.01, format="%.3f",
        help="A score that represents your genetic likelihood of diabetes based on family history. Typical range: 0.078–2.42"
    )
    manual_vals['Age'] = col1.number_input(
        "🎂 Age (years)", 1, 120, 30,
        help="Your current age in years"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Pregnancies = 0 (hidden, sent to model as default)
    final_values = [
        0,  # Pregnancies — default 0, not shown to user
        manual_vals['Glucose'], manual_vals['BloodPressure'],
        manual_vals['SkinThickness'], manual_vals['Insulin'],
        manual_vals['BMI'], manual_vals['DiabetesPedigreeFunction'],
        manual_vals['Age']
    ]

# ─── Paste / NLP Mode ──────────────────────────────────────────────────────────
elif "Paste" in input_mode:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="mode-label">📄 Paste Your Lab Report or Doctor Notes</div>', unsafe_allow_html=True)

    example = """Example:
Patient: Aisha, Age: 34 years
Fasting Blood Sugar (FBS): 145 mg/dL
Blood Pressure: 120/80 mmHg
BMI: 27.5 kg/m2
Insulin: 94 mU/L
Skin Thickness: 29 mm
Diabetes Pedigree: 0.35"""

    st.markdown("""
    <div class="info-box">
    📋 Copy your lab report text and paste it below using <b>Ctrl+V</b> (Windows) or <b>Cmd+V</b> (Mac).
    The AI will automatically extract your health values from the text.
    </div>
    """, unsafe_allow_html=True)

    report_text = st.text_area(
        "Paste your lab report, doctor notes, or any text containing your health values:",
        height=220,
        placeholder=example,
        help="Paste any text — English or mixed Urdu/English. AI will extract all recognizable values automatically."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if report_text.strip():
        extracted_vals, missing = parse_report_text(report_text)

        if extracted_vals:
            st.markdown("**✅ Values extracted from your report:**")
            cols = st.columns(2)
            for i, (k, v) in enumerate(extracted_vals.items()):
                cols[i % 2].success(f"**{FIELD_LABELS.get(k, k)}:** {v}")

        if missing:
            st.warning(f"⚠️ These values were not found. Please enter them manually: **{', '.join([FIELD_LABELS.get(f, f) for f in missing])}**")
            col1, col2 = st.columns(2)
            for i, field in enumerate(missing):
                col = col1 if i % 2 == 0 else col2
                if field in ['BMI', 'DiabetesPedigreeFunction']:
                    extracted_vals[field] = col.number_input(
                        FIELD_LABELS[field], value=FIELD_DEFAULTS[field], step=0.1, format="%.2f"
                    )
                else:
                    extracted_vals[field] = col.number_input(
                        FIELD_LABELS[field], value=int(FIELD_DEFAULTS[field]), step=1
                    )

        if len(extracted_vals) >= 7:
            final_values = [
                0,  # Pregnancies — default 0
                extracted_vals['Glucose'], extracted_vals['BloodPressure'],
                extracted_vals['SkinThickness'], extracted_vals['Insulin'],
                extracted_vals['BMI'], extracted_vals['DiabetesPedigreeFunction'],
                extracted_vals['Age']
            ]

# ─── Predict Button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Diabetes Risk"):
    if final_values is None or len(final_values) < 8:
        st.error("⚠️ Please complete all required values before running the prediction.")
    else:
        errors = []
        if final_values[1] == 0: errors.append("Glucose cannot be 0. Please enter a valid blood sugar value.")
        if final_values[5] == 0: errors.append("BMI cannot be 0. Please enter a valid BMI value.")

        if errors:
            for e in errors:
                st.warning(f"⚠️ {e}")
        else:
            arr = np.array([final_values])
            prediction = model.predict(arr)[0]
            proba = model.predict_proba(arr)[0]
            confidence = round(proba[prediction] * 100, 1)

            st.markdown("---")

            if prediction == 1:
                st.markdown(f"""
                <div class="result-diabetic">
                    <div class="result-icon">🔴</div>
                    <div class="result-title">Diabetes Risk Detected</div>
                    <div class="result-desc">Model Confidence: <strong>{confidence}%</strong></div>
                    <div class="result-desc" style="margin-top:0.5rem;">
                        Please consult your doctor as soon as possible and follow the precautions below.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📋 Recommended Precautions")
                for category, items in DIABETIC_PRECAUTIONS.items():
                    with st.expander(f"{category}", expanded=True):
                        for item in items:
                            st.markdown(f'<div class="precaution-item">• {item}</div>', unsafe_allow_html=True)

                st.error("⚠️ **Important:** This is an AI-based prediction, not a medical diagnosis. Please visit a qualified doctor for a proper evaluation.")

            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div class="result-icon">🟢</div>
                    <div class="result-title">No Diabetes Detected</div>
                    <div class="result-desc">Model Confidence: <strong>{confidence}%</strong></div>
                    <div class="result-desc" style="margin-top:0.5rem;">
                        Great news! No diabetes risk detected — keep up your healthy habits.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 💚 Tips to Stay Healthy")
                for tip in SAFE_TIPS:
                    st.markdown(f'<div class="safe-item">{tip}</div>', unsafe_allow_html=True)

                st.info("ℹ️ Stay consistent with regular checkups, especially if you have a family history of diabetes.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#4a7a8a;font-size:0.8rem;">'
    'DiabetesCare AI &bull; Logistic Regression Model &bull; PIMA Diabetes Dataset<br>'
    '⚕️ For educational purposes only &mdash; always consult a qualified doctor'
    '</div>',
    unsafe_allow_html=True
)
