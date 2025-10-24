import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

# ==========================
# Load Models (AMAN)
# ==========================
@st.cache_resource
def load_models():
    # Load YOLO (pastikan path benar)
    yolo_model = YOLO("model/ica_Laporan4.pt")

    # Load classifier model (keras)
    classifier_path = "model/ica_laporan2.h5"
    classifier = None

    if os.path.exists(classifier_path):
        try:
            # Coba load normal dulu
            classifier = tf.keras.models.load_model(classifier_path, compile=False)
            st.sidebar.success("✅ Model klasifikasi berhasil dimuat.")
        except ValueError as e:
            st.sidebar.warning("⚠️ Model H5 tidak kompatibel. Mencoba konversi dtype...")
            # Fallback: load ulang dengan patch untuk dtype lama
            from tensorflow.keras import models
            import h5py

            with h5py.File(classifier_path, 'r') as f:
                config = f.attrs.get('model_config')
                if config is not None:
                    config = config.decode('utf-8')
                    model = models.model_from_json(config)
                    # hanya muat bobot (tanpa compile)
                    model.load_weights(classifier_path)
                    classifier = model
            st.sidebar.success("✅ Model berhasil direstorasi (fallback mode).")
    else:
        st.sidebar.error("❌ File model klasifikasi tidak ditemukan di folder 'model/'")

    return yolo_model, classifier


# ==========================
# Inisialisasi
# ==========================
yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🔍 Aplikasi Pengenalan Gambar dan Objek")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        if classifier is None:
            st.error("Model klasifikasi tidak berhasil dimuat.")
        else:
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            st.write("### Hasil Prediksi:", class_index)
            st.write("Probabilitas:", f"{confidence:.2%}")
