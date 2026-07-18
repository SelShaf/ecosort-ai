import tensorflow as tf

print("Memuat model H5...")
model = tf.keras.models.load_model("xception_garbage.h5")

# --- TRIK SAKTI: Paksa bungkus model dengan lapisan Softmax baru ---
# Ini untuk memperbaiki lapisan output model H5 kamu yang mungkin kehilangan fungsi aktivasinya
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print("Memulai konversi ke TFLite dengan perbaikan Softmax...")
converter = tf.lite.TFLiteConverter.from_keras_model(probability_model)
tflite_model = converter.convert()

# Simpan berkas hasil konversi baru
with open("xception_garbage.tflite", "wb") as f:
    f.write(tflite_model)

print("Konversi selesai! Silakan buat ulang file .onnx dari file tflite baru ini.")