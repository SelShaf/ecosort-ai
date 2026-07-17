# train_model.py
# Transfer Learning Xception untuk klasifikasi 12 kategori sampah daur ulang
# Setelah training, script ini juga membuat beberapa grafik untuk ditampilkan di halaman web

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Cek versi TensorFlow
print(tf.__version__)

# Folder tujuan penyimpanan grafik
GRAFIK_DIR = "static/img"
os.makedirs(GRAFIK_DIR, exist_ok=True)

# Warna tema, dipakai konsisten di semua grafik
WARNA_UTAMA = "#2E7D4F"
WARNA_SEKUNDER = "#4CAF7D"
WARNA_AKSEN = "#E0985E"

# 1. Load Model Xception Pretrained
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Membekukan semua layer agar tidak di-train ulang (feature extraction)
for layer in base_model.layers:
    layer.trainable = False

# 2. Tambahkan Layer Klasifikasi Baru
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(12, activation='softmax')  # 12 kelas sampah
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Persiapkan Dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'data/train', target_size=(299, 299), batch_size=32, class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    'data/val', target_size=(299, 299), batch_size=32, class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    'data/test', target_size=(299, 299), batch_size=32, class_mode='categorical', shuffle=False
)

class_names = list(train_data.class_indices.keys())
print(train_data.class_indices)

# 4. Latih Model
history = model.fit(train_data, validation_data=val_data, epochs=15)

# 5. Evaluasi Model
test_loss, test_acc = model.evaluate(test_data)
print(f'Akurasi Model: {test_acc:.2f}')

# 6. Simpan model
model.save('xception_garbage.h5')

# ==============================================================
# PEMBUATAN GRAFIK
# ==============================================================

# Grafik 1: Distribusi jumlah gambar per kelas pada dataset asli
kelas_folder = "data/garbage_classification"
jumlah_per_kelas = [
    len(os.listdir(os.path.join(kelas_folder, kelas))) for kelas in class_names
]

plt.figure(figsize=(10, 5))
plt.bar(class_names, jumlah_per_kelas, color=WARNA_UTAMA)
plt.title("Distribusi Jumlah Gambar per Kategori")
plt.xlabel("Kategori")
plt.ylabel("Jumlah Gambar")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_DIR, "distribusi_dataset.png"))
plt.close()

# Grafik 2: Akurasi training vs validation per epoch
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Akurasi Training', color=WARNA_UTAMA)
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi', color=WARNA_AKSEN)
plt.title("Grafik Akurasi Model")
plt.xlabel("Epoch")
plt.ylabel("Akurasi")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_DIR, "akurasi_model.png"))
plt.close()

# Grafik 3: Loss training vs validation per epoch
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Loss Training', color=WARNA_UTAMA)
plt.plot(history.history['val_loss'], label='Loss Validasi', color=WARNA_AKSEN)
plt.title("Grafik Loss Model")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_DIR, "loss_model.png"))
plt.close()

# Grafik 4: Confusion matrix pada data test
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 8))
plt.imshow(cm, cmap="Greens")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Prediksi")
plt.ylabel("Label Sebenarnya")

# Menampilkan angka di tiap sel confusion matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_DIR, "confusion_matrix.png"))
plt.close()

# Cetak laporan klasifikasi lengkap ke terminal
print(classification_report(y_true, y_pred, target_names=class_names))

print("Semua grafik berhasil disimpan di folder static/img/")