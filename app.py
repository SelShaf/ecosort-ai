# app.py
# Aplikasi Flask untuk klasifikasi sampah daur ulang menggunakan Xception

import os
import gc  # Ditambahkan untuk membersihkan RAM secara paksa

# --- OPTIMASI RAM UNTUK RAILWAY (Wajib di bagian paling atas sebelum import TensorFlow) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# Batasi alokasi thread CPU TensorFlow agar tidak menimbun RAM
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# ----------------------------------------------------------------------------------------

from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import gdown

app = Flask(__name__)

# ID file model dari Google Drive kamu
GOOGLE_DRIVE_FILE_ID = "1z3obIRohkucXvCkMv7qzTWjLRx2LeOTI"
MODEL_PATH = "xception_garbage.h5"

# Fungsi pembungkusan load model agar memori unduhan dibersihkan
def init_model():
    if not os.path.exists(MODEL_PATH):
        print("Model belum ada, mengunduh dari Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&confirm=t"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Selesai mengunduh model.")
        
        # Bersihkan sisa buffer dari gdown di RAM sebelum load model TensorFlow
        gc.collect()

    print("Memuat model Xception ke RAM...")
    # Menggunakan tf.keras bawaan tensorflow-cpu 2.15
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
    return loaded_model

# Inisialisasi model
model = init_model()

# Urutan kelas harus sama persis dengan class_indices saat training
class_names = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

non_recyclable = ["trash"]

icon_key = {
    "battery": "battery",
    "biological": "biological",
    "brown-glass": "glass",
    "cardboard": "cardboard",
    "clothes": "clothes",
    "green-glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "shoes": "shoes",
    "trash": "trash",
    "white-glass": "glass"
}

detail_info = {
    "battery": {
        "tipe_material": "Limbah B3 (Bahan Berbahaya dan Beracun)",
        "waktu_terurai": "Hingga 100 tahun",
        "dampak_lingkungan": "Baterai bekas mengandung logam berat yang dapat mencemari tanah dan air tanah jika dibuang sembarangan.",
        "cara_penanganan": "Jangan dibuang ke tempat sampah biasa. Kumpulkan di dropbox baterai bekas atau titik pengumpulan limbah B3 terdekat."
    },
    "biological": {
        "tipe_material": "Sampah Organik",
        "waktu_terurai": "2-4 minggu jika dikompos",
        "dampak_lingkungan": "Jika tercampur dengan sampah lain, sampah organik menghasilkan gas metana di TPA yang berkontribusi pada pemanasan global.",
        "cara_penanganan": "Pisahkan dari sampah lain dan olah menjadi kompos jika memungkinkan."
    },
    "brown-glass": {
        "tipe_material": "Kaca",
        "waktu_terurai": "Lebih dari 1 juta tahun",
        "dampak_lingkungan": "Kaca tidak terurai secara alami, namun dapat didaur ulang berulang kali tanpa kehilangan kualitas.",
        "cara_penanganan": "Bersihkan dari sisa isi, pisahkan dari kaca warna lain sebelum dibuang ke tempat daur ulang kaca."
    },
    "cardboard": {
        "tipe_material": "Kertas Karton",
        "waktu_terurai": "2 bulan",
        "dampak_lingkungan": "Daur ulang kardus membantu mengurangi penebangan pohon untuk bahan baku kertas baru.",
        "cara_penanganan": "Lipat rapi dan pastikan kering sebelum dibuang ke tempat daur ulang kertas."
    },
    "clothes": {
        "tipe_material": "Tekstil",
        "waktu_terurai": "Beberapa bulan hingga puluhan tahun, tergantung bahan",
        "dampak_lingkungan": "Industri tekstil termasuk penyumbang limbah besar, pakaian bekas yang masih layak sebaiknya digunakan kembali.",
        "cara_penanganan": "Sumbangkan jika masih layak pakai, atau salurkan ke bank tekstil untuk didaur ulang."
    },
    "green-glass": {
        "tipe_material": "Kaca",
        "waktu_terurai": "Lebih dari 1 juta tahun",
        "dampak_lingkungan": "Kaca tidak terurai secara alami, namun dapat didaur ulang berulang kali tanpa kehilangan kualitas.",
        "cara_penanganan": "Bersihkan dari sisa isi, pisahkan dari kaca warna lain sebelum dibuang ke tempat daur ulang kaca."
    },
    "metal": {
        "tipe_material": "Logam",
        "waktu_terurai": "50-200 tahun, tergantung jenis logam",
        "dampak_lingkungan": "Daur ulang logam menghemat energi secara signifikan dibanding proses penambangan logam baru.",
        "cara_penanganan": "Bersihkan dari sisa makanan sebelum dibuang ke tempat sampah daur ulang."
    },
    "paper": {
        "tipe_material": "Kertas",
        "waktu_terurai": "2-6 minggu",
        "dampak_lingkungan": "Daur ulang kertas mengurangi kebutuhan penebangan pohon dan konsumsi air dalam produksi kertas baru.",
        "cara_penanganan": "Pastikan kertas tidak basah atau terkontaminasi minyak sebelum didaur ulang."
    },
    "plastic": {
        "tipe_material": "Plastik",
        "waktu_terurai": "50-500 tahun, tergantung jenis plastik",
        "dampak_lingkungan": "Plastik yang tidak dikelola dengan baik berisiko mencemari lingkungan, termasuk perairan dan laut.",
        "cara_penanganan": "Bersihkan dari sisa isi sebelum dibuang ke tempat daur ulang plastik."
    },
    "shoes": {
        "tipe_material": "Campuran (kain, karet, kulit)",
        "waktu_terurai": "25-40 tahun, tergantung bahan",
        "dampak_lingkungan": "Sepatu bekas yang masih layak dapat dipakai kembali sehingga mengurangi limbah tekstil dan karet.",
        "cara_penanganan": "Sumbangkan jika masih layak pakai, atau salurkan ke program daur ulang sepatu."
    },
    "trash": {
        "tipe_material": "Sampah Residu",
        "waktu_terurai": "Bervariasi, umumnya tidak terurai secara alami",
        "dampak_lingkungan": "Sampah residu yang menumpuk di TPA berkontribusi pada pencemaran tanah dan udara.",
        "cara_penanganan": "Buang ke tempat pembuangan akhir, karena kategori ini tidak dapat didaur ulang."
    },
    "white-glass": {
        "tipe_material": "Kaca",
        "waktu_terurai": "Lebih dari 1 juta tahun",
        "dampak_lingkungan": "Kaca tidak terurai secara alami, namun dapat didaur ulang berulang kali tanpa kehilangan kualitas.",
        "cara_penanganan": "Bersihkan dari sisa isi, pisahkan dari kaca warna lain sebelum dibuang ke tempat daur ulang kaca."
    }
}

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((299, 299))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/klasifikasi")
def klasifikasi():
    return render_template("klasifikasi.html")

@app.route("/tentang-dataset")
def tentang_dataset():
    return render_template("tentang_dataset.html")

@app.route("/cara-penggunaan")
def cara_penggunaan():
    return render_template("cara_penggunaan.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0]

    top_index = int(np.argmax(prediction))
    top_label = class_names[top_index]
    top_confidence = float(prediction[top_index]) * 100

    top3_index = np.argsort(prediction)[::-1][:3]
    top3_result = [
        {"label": class_names[i], "confidence": round(float(prediction[i]) * 100, 2)}
        for i in top3_index
    ]

    is_recyclable = top_label not in non_recyclable
    info = detail_info[top_label]

    return jsonify({
        "label": top_label,
        "icon_key": icon_key[top_label],
        "confidence": round(top_confidence, 2),
        "is_recyclable": is_recyclable,
        "tipe_material": info["tipe_material"],
        "waktu_terurai": info["waktu_terurai"],
        "dampak_lingkungan": info["dampak_lingkungan"],
        "cara_penanganan": info["cara_penanganan"],
        "top3": top3_result
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)