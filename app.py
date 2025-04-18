import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# تعيين عنوان الصفحة
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# تخصيص المظهر ليشبه النموذج الأصلي
st.markdown("""
<style>
    .main {
        background-color: #f4f7fa;
    }
    .css-1d391kg, .css-1wrcr25 {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #4CAF50;
        text-align: center;
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 1.2em;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    /* خصائص زر التحديث */
    .refresh-btn {
        background-color: #2196F3 !important;
    }
    .refresh-btn:hover {
        background-color: #0b7dda !important;
    }
    .button-row {
        display: flex;
        gap: 10px;
    }
    .result {
        margin-top: 20px;
        padding: 15px;
        text-align: center;
        font-size: 1.5em;
        color: #333;
        background-color: #e7f3e7;
        border-radius: 5px;
    }
    .error {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

# تحميل النموذج
@st.cache_resource
def load_model():
    try:
        if os.path.exists('diabetes_model.pkl'):
            with open('diabetes_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        else:
            st.error("النموذج غير موجود! الرجاء التأكد من وجود ملف 'diabetes_model.pkl'")
            return None
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل النموذج: {str(e)}")
        return None

# دالة إعادة تعيين النموذج
def reset_form():
    # إعادة تعيين جميع قيم الإدخال عن طريق مسح session state
    for key in list(st.session_state.keys()):
        if key != 'model':  # لا تمسح النموذج المحمّل
            del st.session_state[key]
    
    # تحديث الصفحة - في Streamlit تتم عن طريق rerun
    st.experimental_rerun()

# عنوان التطبيق
st.markdown("<h1>Predict Diabetes</h1>", unsafe_allow_html=True)

# إنشاء النموذج
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1, key="pregnancies")
    glucose = st.number_input("Glucose", min_value=0, step=1, key="glucose")
    blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1, key="blood_pressure")
    skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1, key="skin_thickness")

with col2:
    insulin = st.number_input("Insulin", min_value=0, step=1, key="insulin")
    bmi = st.number_input("BMI", min_value=0.0, step=0.1, format="%.1f", key="bmi")
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, format="%.2f", key="diabetes_pedigree_function")
    age = st.number_input("Age", min_value=0, step=1, key="age")

# أزرار التنبؤ والتحديث
col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("Predict")
with col2:
    # زر التحديث مع تنسيق مخصص
    refresh_button = st.button("Refresh Form", key="refresh", help="إعادة تعيين جميع الحقول")
    # تطبيق تنسيق CSS مخصص على زر التحديث
    st.markdown("""
    <style>
    div[data-testid="stButton"] button:nth-child(3) {
        background-color: #2196F3;
    }
    div[data-testid="stButton"] button:nth-child(3):hover {
        background-color: #0b7dda;
    }
    </style>
    """, unsafe_allow_html=True)

# معالجة زر التحديث
if refresh_button:
    reset_form()

# معالجة زر التنبؤ
if predict_button:
    model = load_model()
    if model:
        try:
            # تجميع القيم في مصفوفة للتنبؤ
            features = [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree_function,
                age
            ]
            
            # التنبؤ باستخدام النموذج
            prediction = model.predict([features])
            
            # عرض النتيجة
            result = int(prediction[0])
            if result == 1:
                st.markdown('<div class="result"><h2>Prediction: Diabetic</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result"><h2>Prediction: Not Diabetic</h2></div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f'<div class="result error"><h2>Error: {str(e)}</h2></div>', unsafe_allow_html=True)
