import os
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

GOOGLE_DRIVE_ONNX_ID = '1XP1NkGxV-cEUVZYaLWoN_HgojKLCWwaq'

def init_model():
    """Fungsi untuk memastikan model ONNX sudah terunduh dan siap digunakan"""
    if not os.path.exists(ONNX_MODEL_PATH):
        print("Model ONNX belum ada, mengunduh dari Google Drive...")
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ONNX_ID}&confirm=t'
        try:
            gdown.download(url, ONNX_MODEL_PATH, quiet=False)
            print("Unduhan model ONNX selesai.")
        except Exception as e:
            print(f"Gagal mengunduh model: {e}")
            raise e

# Jalankan pengecekan/unduhan model saat aplikasi pertama kali dimuat
init_model()

# Inisialisasi ONNX Runtime Session
try:
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name
    print("Model ONNX berhasil dimuat ke ONNX Runtime Session.")
except Exception as e:
    print(f"Gagal memuat ONNX Session: {e}")
    ort_session = None

# Daftar kelas kategori sampah (sesuaikan dengan urutan label saat training)
CLASS_LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

def preprocess_image(image_path):
    """Fungsi untuk memproses gambar agar sesuai dengan input Xception (299x299)"""
    # Baca gambar dalam format BGR
    img = cv2.imread(image_path)
    # Ubah warna dari BGR ke RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize ukuran gambar menjadi 299x299 piksel sesuai arsitektur Xception
    img = cv2.resize(img, (299, 299))
    # Normalisasi piksel ke rentang [0, 1] jika saat training menggunakan preprocess_input Xception standar
    img = img.astype(np.float32) / 255.0
    # Tambahkan dimensi batch: dari (299, 299, 3) menjadi (1, 299, 299, 3)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/klasifikasi', methods=['GET'])
def klasifikasi():
    return render_template('klasifikasi.html')

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
            # Preprocessing gambar
            input_data = preprocess_image(filepath)
            
            # Jalankan prediksi dengan ONNX Runtime
            outputs = ort_session.run(None, {input_name: input_data})
            predictions = outputs[0][0] # Ambil hasil array probabilitas batch pertama
            
            # Cari indeks dengan probabilitas tertinggi
            predicted_idx = int(np.argmax(predictions))
            predicted_label = CLASS_LABELS[predicted_idx]
            confidence = float(predictions[predicted_idx]) * 100

            # (Opsional) Hapus gambar setelah diprediksi untuk menghemat space storage di Railway
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                'class': predicted_label,
                'confidence': f"{confidence:.2f}%"
            })

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Gagal memproses prediksi: {str(e)}'}), 500

if __name__ == '__main__':
    # Untuk local development
    app.run(debug=True, host='0.0.0.0', port=5000)
