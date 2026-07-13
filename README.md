# Rice Quality

## 👥 Kelompok 2

| Nama Anggota                     | NIM              |
| :------------------------------- | :--------------- |
| Hafidz Surya Afifi               | `11230910000002` |
| Ahmad Fauzan Albahy              | `11230910000005` |
| Fahmi Zakaria Nurhasan           | `11230910000053` |
| Muhammad Fikri Rouzan Ash Shidik | `11230910000063` |

---

## 📌 Deskripsi

Implementasi deep learning ini berfungsi untuk mengidentifikasi dan mengklasifikasikan kualitas fisik beras berdasarkan fitur visual pada citra digital. Pemodelan memanfaatkan arsitektur transfer learning MobileNetV2 yang efisien dan ringan untuk melakukan klasifikasi citra secara cepat ke dalam empat kategori mutu fisik. Kategori tersebut meliputi **whole** yaitu biji beras utuh dengan kondisi fisik prima, **chalky** untuk biji beras yang memiliki bintik putih kapur atau berkapur, **broken** untuk kualitas biji beras yang patah atau hancur, serta **discolored** untuk biji beras yang mengalami perubahan warna menjadi kuning atau rusak.

---

## 💾 Dataset

Dataset yang digunakan dalam pengembangan model ini merupakan data primer yang dikumpulkan secara manual dengan total keseluruhan 1.000 citra digital. Seluruh data memiliki distribusi kelas yang seimbang demi menjaga kestabilan performa model saat proses pelatihan, di mana masing-masing kelas memiliki tepat 250 sampel gambar. Kumpulan data ini terbagi rata ke dalam empat kategori kualitas fisik beras yang disesuaikan dengan kondisi riil objek di lapangan, yaitu kategori utuh (whole), kategori berkapur (chalky), kategori patah (broken), dan kategori perubahan warna (discolored).

---

## 🛠️ Tech Stack

| Kategori                    | Teknologi yang Digunakan                                                              |
| :-------------------------- | :------------------------------------------------------------------------------------ |
| 🌐 **Programming Language** | `Python`                                                                              |
| 🌱 **Environment**          | `Jupyter Notebook`                                                                    |
| 🧩 **Frameworks**           | `TensorFlow`, `Streamlit`                                                             |
| ⚛️ **Libraries**            | `NumPy`, `Matplotlib`, `seaborn`, `scikit-learn`, `OpenCV Python`, `Plotly`, `Pillow` |
| ⚡ **Tool**                 | `Google Colab`                                                                        |
| 🚀 **Deployment**           | `Streamlit Community Cloud`                                                           |

---

## ⚙️ Petunjuk Pengaturan

1. **Prasyarat**
   - Python 3.11 atau lebih baru.
   - Git terinstal di komputer.

2. **Clone Repositori**

```bash
git clone https://github.com/Fikri-Rouzan/rice-quality
cd rice-quality
```

3. **Buat Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

4. **Install Dependensi**

```bash
pip install -r requirements.txt
```

5. **Menjalankan Dashboard Streamlit**

```bash
streamlit run dashboard/dashboard.py
```
