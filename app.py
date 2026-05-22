import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="ডায়াবেটিস সনাক্তকরণ অ্যাপ", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .stMetric { background-color: #1c1f26; border-radius: 10px; padding: 15px; }
        #MainMenu { visibility: hidden; }
        header { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)

st.title("🩺 ডায়াবেটিস পূর্বাভাস (Diabetes Prediction)")
c1, c2 = st.columns(2)

with c1:
    pregnancies = st.number_input('গর্ভাবস্থার সংখ্যা (Pregnancies)', 0, 20, 0)
    glucose = st.number_input('গ্লুকোজের মাত্রা (Glucose Level)', 0, 300, 120)
    blood_pressure = st.number_input('রক্তচাপ (Blood Pressure)', 0, 200, 70)
    skin_thickness = st.number_input('ত্বকের পুরুত্ব (Skin Thickness)', 0, 100, 20)

with c2:
    insulin = st.number_input('ইনসুলিনের মাত্রা (Insulin)', 0, 900, 79)
    bmi = st.number_input('বিএমআই (BMI)', 0.0, 70.0, 25.0)
    dpf = st.number_input('ডায়াবেটিস বংশগত স্কোর (Diabetes Pedigree Function)', 0.0, 2.5, 0.5)
    age = st.number_input('বয়স (Age)', 1, 120, 25)

if st.button('ফলাফল দেখুন'):
    data = np.array([[pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    probs = model.predict_proba(data_scaled)[0]

    colR1, colR2 = st.columns(2)
    with colR1:
        st.metric("🔢 ডায়াবেটিস হওয়ার সম্ভাবনা", f"{probs[1]*100:.2f}%")
        st.metric("📉 সুস্থ থাকার সম্ভাবনা", f"{probs[0]*100:.2f}%")

    with colR2:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.bar(['Healthy', 'Diabetic'], probs * 100, color=['green', 'red'])
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probability (%)')
        ax.set_title('Prediction Probability')
        st.pyplot(fig)

    if prediction[0] == 1:
        st.error("⚠️ উক্ত ব্যক্তির ডায়াবেটিস থাকার সম্ভাবনা অনেক বেশি। চিকিৎসকের পরামর্শ নিন।")
    else:
        st.success("✅ উক্ত ব্যক্তির ডায়াবেটিস থাকার সম্ভাবনা নেই বললেই চলে। তিনি সুস্থ আছেন।")