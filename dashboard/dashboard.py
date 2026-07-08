import os

# Membungkam log spam compiler dari TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Ricelytics", page_icon="🌾", layout="wide")

# Definisi konstanta global
LABELS = ["whole", "chalky", "broken", "discolored"]
TARGET_SIZE = (224, 224)
MODEL_PATH = "model/rice_quality.keras"


# Pipeline preprocessing citra
@st.cache_resource
def load_deep_learning_model():
    # Memuat model dengan sistem cache
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Gagal memuat file model. Error: {e}")
            return None
    else:
        st.error(
            f"Model tidak ditemukan di path: {MODEL_PATH}. Pastikan file sudah dipindahkan."
        )
        return None


def enhance_image(image):
    # Mereduksi noise sensor menggunakan Gaussian Blur
    return cv2.GaussianBlur(image, (3, 3), 0)


def preprocess_grain_image(image, target_size=(224, 224)):
    # Segmentasi latar belakang hitam absolut via HSV + Otsu & Resizing
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]

    _, binary_mask = cv2.threshold(
        v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)
    resized_image = cv2.resize(
        segmented_image, target_size, interpolation=cv2.INTER_AREA
    )

    final_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return final_image


# Memuat model MobileNetV2
model = load_deep_learning_model()

# Sidebar panel kontrol
with st.sidebar:
    st.image("https://img.icons8.com/fluency/512/rice-bowl.png", width=120)
    st.title("Ricelytics")
    st.markdown("---")

    st.info("Model: MobileNetV2")

    st.markdown("---")
    st.caption(
        "Dashboard ini dirancang untuk mendeteksi, melakukan segmentasi digital, "
        "dan mengklasifikasikan kualitas bulir beras ke dalam 4 kategori komoditas "
        "menggunakan arsitektur Convolutional Neural Network (CNN)."
    )

# Header dashboard
st.title("Ricelytics: Rice Quality Assessment")
st.markdown(
    "Menampilkan insight segmentasi citra digital dan klasifikasi kualitas beras secara real-time."
)
st.markdown("---")

# Pengaturan tab halaman
tab1, tab2 = st.tabs(["🔍 Quality Inspection", "📖 Kernel Type Guide"])

# Tab 1 untuk inspeksi kualitas beras
with tab1:
    with st.container(border=True):
        st.subheader("Pengaturan Masukan Citra")
        st.markdown(
            "Pilih metode pengambilan atau unggahan foto butir beras di bawah ini:"
        )

        input_method = st.radio(
            "Metode Masukan Gambar:",
            ("Upload File Foto", "Kamera Langsung"),
            horizontal=True,
        )

        st.warning(
            "💡 Rekomendasi: Tempatkan objek beras di dalam lightbox dengan posisi "
            "kamera tegak lurus (90° dari atas objek) demi akurasi visual maksimal."
        )

    uploaded_file = None
    if input_method == "Upload File Foto":
        uploaded_file = st.file_uploader(
            "Unggah foto butir beras (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded_file = st.camera_input(
            "Posisikan objek beras tepat di tengah area kamera"
        )

    # Alur eksekusi pengkondisian citra & inferensi model
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Pemrosesan gambar berurutan
        with st.spinner("Menjalankan preprocessing citra..."):
            enhanced_img = enhance_image(img_bgr)
            processed_img = preprocess_grain_image(
                enhanced_img, target_size=TARGET_SIZE
            )

        # Hasil visual preprocessing
        st.write("")
        st.subheader("🖼️ Analisis Komparasi Citra Digital")
        col1, col2 = st.columns(2)

        with col1:
            st.image(pil_image, caption="Gambar Input Asli", width="stretch")

        with col2:
            st.image(
                processed_img,
                caption="Hasil Segmentasi Latar Belakang Hitam Absolut",
                width="stretch",
            )

        # Proses klasifikasi
        if model is not None:
            st.markdown("---")
            st.subheader("📊 Hasil Analisis Klasifikasi Kualitas")

            # Hitung rasio kepadatan piksel objek
            gray_processed = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            white_pixels = cv2.countNonZero(gray_processed)
            total_pixels = TARGET_SIZE[0] * TARGET_SIZE[1]
            object_ratio = (white_pixels / total_pixels) * 100

            # Deteksi struktur pembatas menggunakan Kontur Bounding Box
            contours, _ = cv2.findContours(
                gray_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            touches_border = False

            if contours:
                # Ambil kontur terbesar
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Cek jika koordinat kotak pelindung menyentuh piksel tepi luar
                if (
                    x <= 2
                    or y <= 2
                    or (x + w) >= (TARGET_SIZE[0] - 2)
                    or (y + h) >= (TARGET_SIZE[1] - 2)
                ):
                    touches_border = True

            # Evaluasi keputusan
            if object_ratio < 2.0 or object_ratio > 40.0 or touches_border:
                st.error("🚨 **ERROR: Invalid Object Detected!**")
                if touches_border:
                    st.write(
                        "**Deteksi Celah Keamanan:** Objek asing terdeteksi memotong atau menyentuh garis pembatas tepi kamera. "
                        "Silakan posisikan ulang butir beras secara mandiri tepat di tengah frame tanpa menyentuh sudut kamera."
                    )
                else:
                    st.write(
                        f"Karakteristik dimensi objek tidak sesuai dengan standar geometri butir beras tunggal "
                        f"(Rasio area aktif: {object_ratio:.2f}%)."
                    )
            else:
                # Proses inferensi
                with st.spinner(
                    "Model MobileNetV2 sedang menganalisis karakteristik piksel..."
                ):
                    normalized_input = processed_img / 255.0
                    input_batch = np.expand_dims(normalized_input, axis=0)

                    predictions = model.predict(input_batch, verbose=0)[0]
                    predicted_class_idx = np.argmax(predictions)
                    predicted_label = LABELS[predicted_class_idx]
                    confidence_score = predictions[predicted_class_idx] * 100

                # Pengaturan visualisasi output berdasarkan label
                if predicted_label == "whole":
                    status_icon = "🌾"
                    alert_text = "Bulir beras terdeteksi berkondisi **Utuh (Whole)** dengan bentuk morfologi sempurna."
                    st.success(
                        f"### {status_icon} KATEGORI: {predicted_label.upper()} ({confidence_score:.2f}%)"
                    )
                elif predicted_label == "chalky":
                    status_icon = "⚪"
                    alert_text = "Bulir beras terdeteksi mengandung kadar kapur tinggi **(Chalky)**, ditandai warna putih susu."
                    st.warning(
                        f"### {status_icon} KATEGORI: {predicted_label.upper()} ({confidence_score:.2f}%)"
                    )
                elif predicted_label == "broken":
                    status_icon = "❌"
                    alert_text = "Bulir beras terdeteksi mengalami patah fisik **(Broken)** yang signifikan di bawah batas normal."
                    st.error(
                        f"### {status_icon} KATEGORI: {predicted_label.upper()} ({confidence_score:.2f}%)"
                    )
                else:
                    status_icon = "🍂"
                    alert_text = "Bulir beras mengalami degradasi atau perubahan warna **(Discolored)** akibat pengaruh eksternal."
                    st.warning(
                        f"### {status_icon} KATEGORI: {predicted_label.upper()} ({confidence_score:.2f}%)"
                    )

                st.write(alert_text)

                # Bar chart distribusi probabilitas
                st.write("")
                st.markdown("#### Grafik Distribusi Probabilitas Prediksi:")

                fig = px.bar(
                    x=[lbl.upper() for lbl in LABELS],
                    y=predictions,
                    labels={
                        "x": "Kategori",
                        "y": "Probabilitas",
                    },
                    color=LABELS,
                    color_discrete_sequence=px.colors.qualitative.Pastel1,
                )

                fig.update_layout(
                    xaxis=dict(
                        tickangle=-45,
                        title_font=dict(size=12),
                    ),
                    yaxis=dict(
                        title_font=dict(size=12),
                        range=[0, 1],
                    ),
                    showlegend=False,
                    height=380,
                    margin=dict(l=40, r=40, t=20, b=60),
                    template="plotly_white",
                )

                st.plotly_chart(fig, width="stretch")

        else:
            st.error("Proses klasifikasi dihentikan karena model gagal dimuat.")
    else:
        st.info(
            "Silakan unggah foto beras atau aktifkan modul kamera untuk memulai proses klasifikasi."
        )

# Tab 2 untuk panduan parameter
with tab2:
    st.subheader("Panduan Standar Mutu Beras")
    st.write(
        "Penjelasan parameter klasifikasi kualitas beras berdasarkan standar karakteristik fisik (kernel):"
    )

    with st.expander("🌾 Whole Kernel (Beras Utuh)"):
        st.markdown("""
        - **Karakteristik**: Bulir beras yang utuh sepenuhnya atau mengalami patah kecil yang tidak melebihi **1/10** bagian dari ukuran panjang rata-rata bulir normal.
        - **Indikator**: Memiliki bentuk morfologi memanjang yang simetris dan tekstur jernih yang dominan.
        """)

    with st.expander("⚪ Chalky Grain (Beras Berkapur)"):
        st.markdown("""
        - **Karakteristik**: Bulir beras yang memiliki tekstur berwarna putih keruh seperti kapur atau susu yang mencakup area **1/2** atau lebih dari keseluruhan tubuh beras.
        - **Indikator**: Kepadatan amilosa yang tidak merata akibat faktor pembentukan endosperma saat fase pematangan tanaman padi.
        """)

    with st.expander("❌ Broken Kernel (Beras Patah)"):
        st.markdown("""
        - **Karakteristik**: Bulir beras yang mengalami kerusakan fisik berupa patahan nyata, dengan ukuran panjang berkisar antara **2/10** hingga **8/10** dari panjang rata-rata bulir utuh.
        - **Indikator**: Terpisahnya bagian kepala atau ekor beras akibat proses penggilingan yang kurang optimal atau tingkat kerapuhan bulir yang tinggi.
        """)

    with st.expander("🍂 Discolored Grain (Beras Berubah Warna)"):
        st.markdown("""
        - **Karakteristik**: Bulir beras yang mengalami perubahan warna secara makro pada sebagian atau seluruh permukaannya, berubah menjadi warna kuning, kuning kecokelatan, hingga bintik hitam.
        - **Indikator**: Kerusakan akibat aktivitas mikroba (jamur), kelembapan box penyimpanan yang buruk, atau efek panas berlebih sebelum proses pengeringan.
        """)

# Footer dashboard
st.markdown("---")
st.caption("© 2026 Ricelytics")
