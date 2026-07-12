import os

# Mengabaikan log spam compiler dari TensorFlow
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
    # Segmentasi latar belakang hitam absolut via Dynamic Max-Relative Thresholding
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]

    # Menghitung nilai ambang batas adaptif dari tingkat kecerahan puncak
    max_v = np.max(v_channel) if np.max(v_channel) > 0 else 1
    dynamic_thresh = max(int(max_v * 0.35), 50)

    _, binary_mask = cv2.threshold(v_channel, dynamic_thresh, 255, cv2.THRESH_BINARY)
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
    st.image("image/icon.png", width=120)
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
    "Menampilkan insight segmentasi citra digital dan klasifikasi kualitas beras."
)
st.markdown("---")

# Pengaturan tab halaman
tab1, tab2 = st.tabs(["🔍 Quality Inspection", "📖 Kernel Type Guide"])

# Tab 1 untuk inspeksi kualitas beras
with tab1:
    with st.container(border=True):
        st.subheader("Pengaturan Masukan Citra")
        st.markdown(
            "Pilih metode pengambilan atau upload foto butir beras di bawah ini:"
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
            "Upload foto butir beras (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded_file = st.camera_input(
            "Posisikan objek beras tepat di tengah area kamera"
        )

    # Alur eksekusi pengkondisian citra & inferensi model objek
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Pemrosesan gambar sekuensial penuh pada resolusi asli
        with st.spinner("Menjalankan preprocessing dan lokalisasi objek..."):
            enhanced_img = enhance_image(img_bgr)

            # Segmentasi penuh untuk memisahkan latar belakang secara global
            hsv_image = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
            v_channel = hsv_image[:, :, 2]

            # Sinkronisasi parameter binarisasi
            max_v_main = np.max(v_channel) if np.max(v_channel) > 0 else 1
            dynamic_thresh_main = max(int(max_v_main * 0.35), 50)

            _, binary_mask = cv2.threshold(
                v_channel, dynamic_thresh_main, 255, cv2.THRESH_BINARY
            )

            # Menemukan seluruh kontur independen butir beras di dalam gambar
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

        st.write("")
        st.subheader("🖼️ Analisis Komparasi Dan Hasil Deteksi")

        # Inisialisasi variabel penghitung dan penanda objek beras
        grain_counts = {"whole": 0, "chalky": 0, "broken": 0, "discolored": 0}
        detected_any_rice = False

        # Proses ekstraksi, pembersihan latar belakang, dan inferensi model
        if model is not None:
            if contours:
                # Mencari kontur terbesar di layar sebagai acuan skala relatif
                largest_contour_main = max(contours, key=cv2.contourArea)
                max_grain_area = cv2.contourArea(largest_contour_main)

                # Pembuatan mask murni
                clean_full_mask = np.zeros_like(binary_mask)
                valid_grains_data = []

                # Tahap filter 1 untuk validasi kontur morfologi beras
                for idx, c in enumerate(contours):
                    area = cv2.contourArea(c)

                    # Filter dimensi dasar geometri beras
                    if area < 100:
                        continue
                    if area < (max_grain_area * 0.10):
                        continue
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    solidity_score = float(area) / hull_area if hull_area > 0 else 0
                    if solidity_score < 0.75:
                        continue

                    x, y, w, h = cv2.boundingRect(c)
                    aspect_ratio_score = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                    if aspect_ratio_score < 1.10:
                        continue

                    # Objek dinyatakan lolos sebagai komponen beras valid
                    detected_any_rice = True
                    cv2.drawContours(
                        clean_full_mask, [c], -1, 255, thickness=cv2.FILLED
                    )
                    valid_grains_data.append((x, y, w, h))

                # Terapkan mask untuk menghasilkan citra segmentasi bebas noise
                segmented_clean_bgr = cv2.bitwise_and(
                    img_bgr, img_bgr, mask=clean_full_mask
                )
                img_rgb_annotated = cv2.cvtColor(segmented_clean_bgr, cv2.COLOR_BGR2RGB)
                segmented_full_rgb = img_rgb_annotated.copy()

                # Tahap filter 2 untuk proses prediksi dengan square padding adaptif
                for x, y, w, h in valid_grains_data:
                    # Potong objek bulir beras murni
                    grain_crop = segmented_clean_bgr[y : y + h, x : x + w]

                    # Membuat kanvas kotak murni 1:1 berdasarkan sisi terpanjang
                    max_side = max(w, h)
                    grain_square = np.zeros((max_side, max_side, 3), dtype=np.uint8)
                    df_x = (max_side - w) // 2
                    df_y = (max_side - h) // 2
                    grain_square[df_y : df_y + h, df_x : df_x + w] = grain_crop

                    # Resize kanvas kotak proporsional ke skala input model
                    grain_resized = cv2.resize(
                        grain_square, TARGET_SIZE, interpolation=cv2.INTER_AREA
                    )
                    grain_input_rgb = cv2.cvtColor(grain_resized, cv2.COLOR_BGR2RGB)

                    # Normalisasi dan inferensi model secara terlokalisasi
                    normalized_input = grain_input_rgb / 255.0
                    input_batch = np.expand_dims(normalized_input, axis=0)
                    predictions = model.predict(input_batch, verbose=0)[0]

                    predicted_class_idx = np.argmax(predictions)
                    predicted_label = LABELS[predicted_class_idx]
                    confidence_score = predictions[predicted_class_idx] * 100

                    # Memperbarui kamus jumlah batch kualitas beras
                    grain_counts[predicted_label] += 1

                    # Pemetaan warna kotak pembatas berdasarkan kategori
                    color_map = {
                        "whole": (0, 255, 0),
                        "chalky": (255, 165, 0),
                        "broken": (255, 0, 0),
                        "discolored": (255, 255, 0),
                    }
                    box_color = color_map[predicted_label]

                    # Mengatur skala font teks label secara dinamis
                    dynamic_font_scale = max(0.35, img_bgr.shape[1] / 3200.0)
                    dynamic_thickness = max(1, int(img_bgr.shape[1] / 2200.0))

                    # Kotak pembatas pada citra berlatar belakang hitam
                    cv2.rectangle(
                        img_rgb_annotated,
                        (x, y),
                        (x + w, y + h),
                        box_color,
                        dynamic_thickness + 1,
                    )
                    label_text = f"{predicted_label.upper()} ({confidence_score:.0f}%)"
                    cv2.putText(
                        img_rgb_annotated,
                        label_text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        dynamic_font_scale,
                        box_color,
                        dynamic_thickness,
                    )

            # Visualisasi di kolom dashboard
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    pil_image,
                    caption="Gambar Input Asli",
                    width="stretch",
                )
            with col2:
                if detected_any_rice:
                    st.image(
                        img_rgb_annotated,
                        caption="Hasil Segmentasi",
                        width="stretch",
                    )
                else:
                    st.image(
                        segmented_full_rgb,
                        caption="Hasil Segmentasi (Tidak Ada Objek Beras Valid)",
                        width="stretch",
                    )

            # Statistik kuantitatif kumulatif
            if detected_any_rice:
                st.markdown("---")
                st.subheader("📊 Hasil Analisis Kuantitas Komoditas Beras")

                # Menghitung total jumlah butir beras dari semua kategori
                total_grains = sum(grain_counts.values())

                st.markdown(f"🎯 **Total Seluruh Beras:** `{total_grains}`")
                st.write("")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        label="🌾 TOTAL UTUH (WHOLE)", value=grain_counts["whole"]
                    )
                with m2:
                    st.metric(
                        label="⚪ TOTAL BERKAPUR (CHALKY)",
                        value=grain_counts["chalky"],
                    )
                with m3:
                    st.metric(
                        label="❌ TOTAL PATAH (BROKEN)",
                        value=grain_counts["broken"],
                    )
                with m4:
                    st.metric(
                        label="🍂 TOTAL BERUBAH WARNA (DISCOLORED)",
                        value=grain_counts["discolored"],
                    )

                # Bar chart akumulasi jumlah butir per label
                st.write("")
                st.markdown("#### Grafik Akumulasi Distribusi Jumlah Butir Beras:")

                categories = [lbl for lbl in LABELS]
                total_counts = [grain_counts[lbl] for lbl in LABELS]

                fig = px.bar(
                    x=categories,
                    y=total_counts,
                    labels={
                        "x": "Kategori Kualitas Beras",
                        "y": "Jumlah Butir",
                    },
                    color=LABELS,
                    color_discrete_sequence=px.colors.qualitative.Pastel1,
                    text=total_counts,
                )
                fig.update_traces(textposition="auto")
                fig.update_layout(
                    xaxis=dict(tickangle=-45, title_font=dict(size=12)),
                    yaxis=dict(title_font=dict(size=12)),
                    showlegend=False,
                    height=380,
                    margin=dict(l=40, r=40, t=20, b=60),
                    template="plotly_white",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.error(
                    "🚨 **Validasi Gagal:** Tidak ada objek butir beras yang memenuhi standar kriteria morfologi geometri sistem."
                )
        else:
            st.error("Proses klasifikasi dihentikan karena model gagal dimuat.")
    else:
        st.info(
            "Silakan upload foto beras atau aktifkan modul kamera untuk memulai proses klasifikasi."
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

# Footer
st.markdown("---")
st.caption("© 2026 Ricelytics")
