# EcoSort AI - Klasifikasi Sampah Daur Ulang Berbasis Web

Aplikasi web berbasis Flask yang menggunakan Transfer Learning dengan arsitektur Xception untuk mengklasifikasikan sampah rumah tangga ke dalam 12 kategori, guna mempermudah proses pemilahan sampah untuk daur ulang.

## Tentang Project

EcoSort AI dikembangkan sebagai tugas mata kuliah Kecerdasan Buatan, dengan menerapkan konsep Transfer Learning menggunakan model Xception yang telah dilatih sebelumnya pada dataset ImageNet. Model ini kemudian disesuaikan (fine-tuned) untuk mengenali 12 kategori sampah rumah tangga dari foto yang diunggah pengguna.

Selain memberikan hasil klasifikasi, aplikasi ini juga menampilkan informasi tambahan seperti tipe material, estimasi waktu terurai, dampak lingkungan, dan cara penanganan yang tepat untuk setiap kategori sampah yang terdeteksi.

## Fitur

- Klasifikasi gambar sampah ke dalam 12 kategori menggunakan model Xception
- Upload gambar dengan drag and drop atau klik untuk memilih file
- Hasil analisis lengkap: tingkat keyakinan (confidence score), tipe material, estimasi waktu terurai, dampak lingkungan, dan cara penanganan
- Menampilkan 3 kemungkinan kategori teratas beserta persentasenya
- Halaman informasi dataset lengkap dengan grafik distribusi data dan performa model (akurasi, loss, confusion matrix)
- Halaman panduan penggunaan dengan FAQ
- Desain responsif untuk perangkat mobile dan desktop
- Tema visual gelap (dark eco theme) dengan ikon SVG custom

## Teknologi yang Digunakan

**Machine Learning**
- TensorFlow / Keras
- Transfer Learning dengan arsitektur Xception (pretrained ImageNet)
- TensorFlow Lite untuk optimasi ukuran model saat deployment

**Backend**
- Python 3.11
- Flask

**Frontend**
- HTML5, CSS3 (custom, tanpa framework)
- JavaScript (vanilla)
- Font: Poppins (Google Fonts)

**Tools Pendukung**
- Matplotlib dan Scikit-learn (untuk visualisasi hasil training)
- split-folders (untuk membagi dataset)
- gdown (untuk mengunduh model dari Google Drive saat deployment)

**Deployment**
- GitHub (version control)
- Railway (hosting aplikasi)
- Google Drive (penyimpanan file model)

## Dataset

Dataset yang digunakan adalah **Garbage Classification** dari Kaggle, dikumpulkan oleh Mostafa Abla.

- Sumber: [Kaggle - Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- Jumlah gambar: sekitar 15.500 gambar
- Jumlah kelas: 12 kategori
  - battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass
- Pembagian data: 70% training, 15% validasi, 15% testing

**Keterbatasan dataset:** sebagian gambar berasal dari web scraping, bukan foto asli sampah dari tempat pembuangan, sehingga akurasi di dunia nyata mungkin sedikit berbeda dari hasil pengujian.

## Struktur Folder
PROJECT_XCEPTION_GARBAGE/
├── venv/                       (virtual environment, tidak masuk repo)
├── data/
│   ├── garbage_classification/ (dataset mentah, tidak masuk repo)
│   ├── train/
│   ├── val/
│   └── test/
├── templates/
│   ├── index.html
│   ├── klasifikasi.html
│   ├── tentang_dataset.html
│   └── cara_penggunaan.html
├── static/
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── script.js
│   ├── images/                 (foto contoh tiap kategori sampah)
│   └── img/                    (grafik hasil training)
├── split_dataset.py
├── train_model.py
├── convert.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

## Instalasi dan Menjalankan Secara Lokal

### Prasyarat
- Python 3.11
- pip
- Git

### Langkah-langkah

1. Clone repository ini
git clone https://github.com/SelsaShaf/ecosort-ai.git
cd ecosort-ai

2. Buat dan aktifkan virtual environment
python -m venv venv
Windows:
venv\Scripts\activate
macOS/Linux:
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Jalankan aplikasi
python app.py
Model akan otomatis terunduh dari Google Drive saat aplikasi pertama kali dijalankan.

5. Buka browser ke `http://127.0.0.1:5000`

## Training Model

Jika ingin melatih ulang model dari awal:

1. Download dataset dari Kaggle dan ekstrak ke folder `data/garbage_classification/`

2. Jalankan script pembagian dataset
python split_dataset.py

3. Jalankan script training
python train_model.py
Script ini akan menghasilkan file model dan otomatis membuat grafik distribusi dataset, akurasi, loss, serta confusion matrix di folder `static/img/`.

4. Jika ukuran model terlalu besar untuk deployment, jalankan konversi ke TensorFlow Lite
python convert.py

**Catatan:** proses training menggunakan CPU biasa bisa memakan waktu lama. Disarankan menggunakan Google Colab dengan GPU untuk mempercepat proses.

### Spesifikasi Model
- Arsitektur dasar: Xception (pretrained ImageNet, layer dibekukan)
- Layer tambahan: GlobalAveragePooling2D, Dense(256, relu), Dropout(0.5), Dense(12, softmax)
- Ukuran input gambar: 299 x 299 piksel
- Loss function: categorical_crossentropy
- Optimizer: Adam

## Deployment

Model tidak disertakan langsung di repository GitHub karena ukurannya besar. Model diunggah secara terpisah ke Google Drive, dan aplikasi (`app.py`) akan mengunduhnya secara otomatis saat pertama kali dijalankan menggunakan library `gdown`.

Alur deployment:
1. Kode aplikasi di-push ke GitHub (tanpa file model)
2. File model diunggah ke Google Drive dengan akses publik
3. Aplikasi di-deploy ke Railway, terhubung langsung ke repository GitHub
4. Railway otomatis build dan menjalankan aplikasi menggunakan `requirements.txt`
5. Saat aplikasi pertama kali start, model otomatis terunduh dari Google Drive

Variabel yang perlu disesuaikan sebelum deploy:
- `GOOGLE_DRIVE_FILE_ID` di dalam `app.py`, isi dengan ID file model dari Google Drive

## Cara Penggunaan Aplikasi

1. Buka halaman "Klasifikasi" melalui menu navigasi
2. Unggah foto sampah dengan cara drag and drop atau klik area upload
3. Pastikan foto jelas dan hanya menampilkan satu jenis sampah
4. Klik tombol "Analisis Gambar"
5. Hasil klasifikasi akan ditampilkan lengkap dengan tingkat keyakinan, tipe material, dampak lingkungan, dan cara penanganan yang disarankan

## Performa Model

Hasil evaluasi model dapat dilihat pada halaman "Tentang Dataset" di aplikasi, meliputi:
- Grafik distribusi jumlah gambar per kategori
- Grafik akurasi training dan validasi per epoch
- Grafik loss training dan validasi per epoch
- Confusion matrix pada data test

## Kontributor

Dikembangkan oleh Selsa Shafana Alfiyani (NIM 301240041) sebagai tugas mata kuliah Kecerdasan Buatan.