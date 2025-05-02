import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import kagglehub

# Unduh dataset
path = kagglehub.dataset_download("steve1215rogg/student-lifestyle-dataset")

# Load scaler, encoder, dan model
scaler = joblib.load(f"{path}/scaler.pkl")
label_encoder = joblib.load(f"{path}/label_encoder.pkl")
interpreter = tf.lite.Interpreter(model_path=f"{path}/student_lifestyle.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Kesehatan Mahasiswa")
st.write("Masukkan informasi gaya hidup untuk memprediksi tingkat kesehatan atau kebugaran.")

# Input pengguna
sleep_hours = st.slider("Jam tidur per hari", 0, 12, 7)
study_hours = st.slider("Jam belajar per hari", 0, 12, 4)
exercise_hours = st.slider("Jam olahraga per minggu", 0, 20, 3)
screen_time = st.slider("Waktu layar per hari (jam)", 0, 15, 6)
junk_food = st.selectbox("Konsumsi junk food", ["Sering", "Kadang", "Jarang"])

# Konversi kategorik
junk_food_mapping = {"Sering": 2, "Kadang": 1, "Jarang": 0}
junk_food_val = junk_food_mapping[junk_food]

# Prediksi saat tombol ditekan
if st.button("Prediksi Kesehatan"):
    input_data = np.array([[sleep_hours, study_hours, exercise_hours, screen_time, junk_food_val]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = np.argmax(prediction)
    health_status = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Prediksi tingkat kesehatan: **{health_status.upper()}**")
