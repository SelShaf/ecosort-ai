# split_dataset.py
# Membagi dataset garbage_classification menjadi train/val/test
# Rasio: 70% train, 15% validation, 15% test

import splitfolders

input_folder = "data/garbage_classification"
output_folder = "data"

splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=42,
    ratio=(0.7, 0.15, 0.15)
)

print("Selesai! Struktur folder data/train, data/val, data/test sudah siap.")