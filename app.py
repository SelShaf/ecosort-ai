import os
import gc 

# --- OPTIMASI RAM UNTUK RAILWAY (Dipertahankan dari versi lama) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import flask
from flask import Flask, request, jsonify, render_template
import werkzeug
import numpy as np
import cv2
import onnxruntime as ort
import gdown

app = Flask(__name__)

# --- KONFIGURASI DIREKTORI & MODEL ---
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ONNX_MODEL_PATH = 'xception_garbage.onnx'
# Menggunakan Google Drive ID milik Selsa dari versi lama
GOOGLE_DRIVE_ONNX_ID = '1XP1NkGxV-cEUVZYaLWoN_HgojKLCWwaq'

ort_session = None

def init_model():
    """Fungsi untuk memastikan model ONNX sudah terunduh dan siap digunakan"""
    global ort_session
    if not os.path.exists(ONNX_MODEL_PATH):
        print("Model ONNX belum ada, mengunduh dari Google Drive...")
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ONNX_ID}&confirm=t'
        try:
            gdown.download(url, ONNX_MODEL_PATH, quiet=False)
            print("Unduhan model ONNX selesai.")
            gc.collect()
        except Exception as e:
            print(f"Gagal mengunduh model: {e}")
            raise e

    print("Memuat model Xception via ONNX Runtime ke RAM...")
    try:
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
        print("Model ONNX berhasil dimuat ke ONNX Runtime Session!")
    except Exception as e:
        print(f"Gagal memuat ONNX Session: {e}")
        ort_session = None

# Jalankan inisialisasi model saat startup aplikasi
init_model()

# --- DATASET KELAS & DETAIL INFORMASI (Lengkap dari versi lama) ---
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
        "waktu_terurai": "2-5 bulan",
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

def preprocess_image(image_path):
    """Fungsi preprocessing gambar, disesuaikan agar sama persis dengan normalisasi saat training"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))

    # Normalisasi harus sama dengan rescale=1./255 yang dipakai di train_model.py
    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    return img

# ================= RUTE HALAMAN WEB =================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')

@app.route('/tentang-dataset')
def tentang_dataset():
    return render_template('tentang_dataset.html')

@app.route('/cara-penggunaan')
def cara_penggunaan():
    return render_template('cara_penggunaan.html')

# ================= RUTE API PREDIKSI (Disinkronkan Total) =================

@app.route('/predict', methods=['POST'])
def predict():
    if ort_session is None:
        return jsonify({'error': 'Model tidak tersedia di server'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang dikirim'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    if file:
        filename = werkzeug.utils.secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            input_data = preprocess_image(filepath)
            
            # Jalankan prediksi dengan ONNX Runtime
            input_name = ort_session.get_inputs()[0].name
            prediction = ort_session.run(None, {input_name: input_data})[0].flatten()
            
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

            # Hapus berkas setelah selesai diproses demi hemat storage RAM
            if os.path.exists(filepath):
                os.remove(filepath)

            # Mengembalikan format JSON lengkap agar dibaca sempurna oleh HTML/JS lama
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

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Gagal memproses prediksi: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
