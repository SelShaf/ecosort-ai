import tensorflow as tf

# 1. Muat model h5 kamu yang ada di laptop
model = tf.keras.models.load_model("xception_garbage.h5")

# 2. Konversi ke format TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Opsional: Aktifkan optimasi ukuran (kuantisasi) agar makin ringan
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 3. Simpan berkas hasil konversi
with open("xception_garbage.tflite", "wb") as f:
    f.write(tflite_model)

print("Konversi selesai! Berkas xception_garbage.tflite berhasil dibuat.")