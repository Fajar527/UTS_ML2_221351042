import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="student_lifestyle.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Student Lifestyle")
st.write("masukan data mahasiswa")

# Form input pengguna (disesuaikan dengan dataset student lifestyle)
sleep = st.number_input("Jam tidur per hari", min_value=0.0, max_value=12.0, value=7.0)
study = st.number_input("Jam belajar per hari", min_value=0.0, max_value=12.0, value=4.0)
exercise = st.number_input("Jam olahraga per minggu", min_value=0.0, max_value=20.0, value=3.0)
screen_time = st.number_input("Waktu layar per hari (jam)", min_value=0.0, max_value=15.0, value=6.0)
junk_food = st.selectbox("Frekuensi konsumsi junk food", ["Jarang", "Kadang", "Sering"])
GPA = st.number_input("GPA", min_value=0.0, max_value=20.0, value=3.0)

# Konversi nilai kategorikal ke numerik
junk_food_mapping = {"Jarang": 0, "Kadang": 1, "Sering": 2}
junk_food_val = junk_food_mapping[junk_food]

if st.button("Prediksi Status Kesehatan"):
    # Preprocessing input
    input_data = np.array([[sleep, study, exercise, screen_time, junk_food_val, GPA]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    health_status = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Prediksi status kesehatan: **{health_status.upper()}**")

