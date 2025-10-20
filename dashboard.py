pip install streamlit
pip install ultralytics
pip install tensorflow
pip install pillow
pip install opencv-python
pip install numpy

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/best.pt")  # Model deteksi objek
        classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
        return yolo_model, classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

yolo_model, classifier = load_models()

# Daftar kelas untuk klasifikasi (sesuaikan dengan model Anda)
class_names = ['Class A', 'Class B', 'Class C']  # Ganti dengan nama kelas aktual Anda

# ==========================
# Custom CSS untuk Tema Warna Spectral (Pelangi)
# ==========================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #ff6b6b, #ffa500, #ffff00, #32cd32, #1e90ff, #9370db, #8a2be2);
        background-attachment: fixed;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff4500, #ff6347);
        color: white;
        border-radius: 15px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 6px 12px rgba(255, 69, 0, 0.3);
        transition: all 0.4s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #ff6347, #ff4500);
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(255, 69, 0, 0.5);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #9370db, #8a2be2);
        color: #ffffff;
        border-radius: 10px;
        padding: 20px;
    }
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border: 3px solid #1e90ff;
    }
    .prediction-box {
        background: linear-gradient(135deg, #32cd32, #1e90ff);
        padding: 20px;
        border-radius: 15px;
        border-left: 8px solid #ffff00;
        margin-top: 25px;
        color: #ffffff;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .header-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(45deg, #ff6b6b, #ffa500, #ffff00, #32cd32, #1e90ff, #9370db, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        background: linear-gradient(135deg, #8a2be2, #ff6b6b);
        padding: 15px;
        border-radius: 10px;
        color: #ffffff;
        margin-top: 30px;
    }
    .profile-img {
        border-radius: 50%;
        border: 4px solid #ffff00;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        width: 100px;
        height: 100px;
        object-fit: cover;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# UI Header dengan Tema Spectral
# ==========================
st.markdown('<div class="header-title">🌈 Image Classification & Object Detection App</div>', unsafe_allow_html=True)
st.markdown("### 🚀 Unggah gambar dan pilih mode untuk analisis cerdas menggunakan AI dengan tema pelangi yang menawan!")
st.markdown("---")

# ==========================
# Sidebar Menu dengan Tema Spectral
# ==========================
st.sidebar.title("⚙️ Pengaturan")
menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"],
    help="Pilih mode analisis gambar yang diinginkan."
)

# Opsi tambahan di sidebar
st.sidebar.markdown("### 🌟 Opsi Tambahan")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (YOLO)",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Atur ambang batas kepercayaan untuk deteksi objek."
)

# ==========================
# File Uploader
# ==========================
uploaded_file = st.file_uploader(
    "📤 Unggah Gambar",
    type=["jpg", "jpeg", "png"],
    help="Pilih file gambar untuk dianalisis."
)

if uploaded_file is not None:
    # Load dan tampilkan gambar
    img = Image.open(uploaded_file)
    
    # Layout dengan kolom
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🖼️ Gambar Asli")
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)
    
    with col2:
        if menu == "Deteksi Objek (YOLO)":
            if yolo_model is None:
                st.error("Model YOLO tidak dapat dimuat.")
            else:
                st.markdown("### 🔍 Hasil Deteksi Objek")
                with st.spinner("Mendeteksi objek..."):
                    # Deteksi objek dengan threshold
                    results = yolo_model(img, conf=confidence_threshold)
                    result_img = results[0].plot()  # Gambar dengan bounding box
                    
                    # Tampilkan hasil
                    st.image(result_img, caption="Hasil Deteksi dengan Bounding Box", use_container_width=True)
                    
                    # Tampilkan detail deteksi
                    detections = results[0].boxes
                    if len(detections) > 0:
                        st.markdown("#### 📋 Detail Deteksi:")
                        for i, box in enumerate(detections):
                            cls = int(box.cls[0])
                            conf = box.conf[0]
                            st.write(f"- 🌟 Objek {i+1}: Kelas {cls}, Confidence: {conf:.2f}")
                    else:
                        st.info("Tidak ada objek terdeteksi dengan threshold yang dipilih.")
        
        elif menu == "Klasifikasi Gambar":
            if classifier is None:
                st.error("Model klasifikasi tidak dapat dimuat.")
            else:
                st.markdown("### 🧠 Hasil Klasifikasi")
                with st.spinner("Mengklasifikasikan gambar..."):
                    # Preprocessing
                    img_resized = img.resize((224, 224))  # Sesuaikan ukuran dengan model Anda
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0
                    
                    # Prediksi
                    prediction = classifier.predict(img_array)
                    class_index = np.argmax(prediction)
                    class_name = class_names[class_index] if class_index < len(class_names) else f"Class {class_index}"
                    confidence = np.max(prediction)
                    
                    # Tampilkan hasil dalam box yang menarik
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h4>🎯 Prediksi: {class_name}</h4>
                        <p><strong>Confidence:</strong> {confidence:.2f}</p>
                        <p><strong>Probabilitas untuk semua kelas:</strong></p>
                        <ul>
                    """, unsafe_allow_html=True)
                    
                    for i, prob in enumerate(prediction[0]):
                        class_label = class_names[i] if i < len(class_names) else f"Class {i}"
                        st.markdown(f"<li>🌈 {class_label}: {prob:.2f}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    # Tambahkan bar chart untuk probabilitas
                    st.markdown("#### 📊 Distribusi Probabilitas")
                    prob_dict = {class_names[i] if i < len(class_names) else f"Class {i}": float(prob) for i, prob in enumerate(prediction[0])}
                    st.bar_chart(prob_dict)

# ==========================
# Footer dengan Foto Profil Developer
# ==========================
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
col_footer1, col_footer2 = st.columns([1, 3])
with col_footer1:
    # Asumsikan ada file "developer_profile.jpg" di folder yang sama; ganti dengan path Anda
    try:
        st.image("developer_profile.jpg", caption="👨‍💻 Developer", use_container_width=False, width=100)
    except:
        st.markdown("👨‍💻 [Foto Profil Developer]")  # Placeholder jika file tidak ada
with col_footer2:
    st.markdown("**Dibuat dengan ❤️ menggunakan Streamlit, YOLOv8, dan TensorFlow.**")
    st.markdown("Untuk pertanyaan atau dukungan, hubungi [developer@example.com](mailto:developer@example.com).")
st.markdown('</div>', unsafe_allow_html=True)