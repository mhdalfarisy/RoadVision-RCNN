# 🚗 Deteksi Kendaraan dengan Faster R-CNN MobileNetV3

Deteksi objek kendaraan (mobil, bus, truk, sepeda motor, dan ambulans) menggunakan **Faster R-CNN dengan backbone MobileNetV3-Large 320 FPN**, dilatih di Google Colab (GPU T4) menggunakan dataset dari Roboflow dan dievaluasi dengan metrik standar COCO.

Proyek ini dibuat untuk tugas **UTS Computer Vision Advance** (UNPAM MTI REG C).

![PyTorch](https://img.shields.io/badge/PyTorch-Torchvision-EE4C2C?logo=pytorch&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?logo=googlecolab&logoColor=white)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-6706CE)

---

## 📌 Ringkasan Hasil

| Metrik | Validasi (100 gambar) | Test |
| --- | --- | --- |
| mAP@[0.5:0.95] | 0.4163 | **0.4287** |
| mAP@0.5 | 0.6216 | **0.6348** |
| AP@0.75 | 0.448 | 0.488 |
| AP objek besar | 0.461 | 0.488 |
| AP objek sedang | 0.020 | 0.145 |
| AP objek kecil | 0.000 | 0.000 |

Hasil di atas dicapai hanya dengan **300 gambar latih dan 3 epoch**, sehingga cocok sebagai eksperimen cepat untuk menguji alur kerja.

---

## 🧭 Alur Kerja

1. Instalasi pustaka pendukung
2. Impor pustaka dan penetapan hyperparameter
3. Pengunduhan dataset dari Roboflow (format COCO)
4. Pembuatan `Dataset` dan `DataLoader` kustom
5. Pembangunan model Faster R-CNN
6. Pelatihan dengan evaluasi mAP di setiap epoch
7. Evaluasi model terbaik pada data validasi dan data uji
8. Visualisasi prediksi dan uji coba pada gambar baru

---

## 🗂️ Dataset

- **Sumber:** Roboflow, diunduh dalam format **COCO** (terbagi otomatis menjadi `train`, `valid`, dan `test`).
- **Kelas:** `Ambulance`, `Bus`, `Car`, `Motorcycle`, `Truck`, ditambah satu kategori bawaan Roboflow (`cars`, ID 0).
- **Jumlah kelas model:** 6 (`max(cat_ids) + 1`, karena indeks 0 dipakai sebagai *background* oleh Faster R-CNN).
- **Subset yang dipakai:** 300 gambar latih dan 100 gambar validasi (data uji dipakai seluruhnya).

---

## 🧠 Model

- **Arsitektur:** `fasterrcnn_mobilenet_v3_large_320_fpn` dari Torchvision
- **Transfer learning:** bobot pra-latih COCO, lalu lapisan akhir `FastRCNNPredictor` diganti sesuai jumlah kelas dataset
- **Optimizer:** SGD (momentum 0.9, weight decay 0.0005)
- **Scheduler:** StepLR (`step_size=2`, `gamma=0.1`)
- **Checkpoint:** bobot disimpan ke `rcnn_best_model.pth` hanya ketika mAP@[0.5:0.95] validasi membaik

### Hyperparameter

| Parameter | Nilai |
| --- | --- |
| Jumlah gambar latih | 300 |
| Jumlah gambar validasi | 100 |
| Batch size | 4 |
| Jumlah epoch | 3 |
| Learning rate | 0.005 |
| Seed | 42 |
| Freeze backbone | `False` |

---

## 📈 Hasil Pelatihan

| Epoch | Rata-rata loss | mAP@[0.5:0.95] | mAP@0.5 |
| --- | --- | --- | --- |
| 1 | 1.2552 | 0.2743 | 0.488 |
| 2 | 0.9488 | 0.3670 | 0.560 |
| 3 | 0.7362 | 0.4163 | 0.622 |

Loss turun konsisten dan mAP validasi naik di setiap epoch tanpa tanda *overfitting* dalam 3 epoch. Karena mAP masih naik di epoch terakhir, model kemungkinan masih bisa membaik jika dilatih lebih lama.

<!-- Ganti path gambar sesuai lokasi di repositori Anda -->
![Kurva training loss dan validation mAP](images/kurva_training.png)

*Kurva training loss (kiri) dan validation mAP (kanan) per epoch.*

---

## 🖼️ Contoh Prediksi

### Data validasi (ambang skor 0.5)

![Prediksi pada data validasi](images/prediksi_validasi.png)

Kotak pembatas umumnya menempel rapat pada objek utama. Masih ada kesalahan klasifikasi, misalnya sebuah mobil putih yang diberi label *Ambulance* dengan skor tipis 0.55, kemungkinan karena data ambulans yang sedikit.

### Uji coba pada foto lalu lintas perkotaan

![Hasil deteksi pada foto lalu lintas](images/uji_coba_lalu_lintas.png)

Model menemukan 5 objek dengan skor di atas 0.5 (empat mobil dan satu sepeda motor) tanpa kotak ganda pada objek yang sama.

---

## ⚠️ Keterbatasan

- **Objek kecil sulit terdeteksi** (AP objek kecil = 0). Input diperkecil ke sekitar 320 piksel demi kecepatan sehingga detail objek kecil banyak yang hilang.
- **Sepeda motor pada kerumunan padat banyak terlewat**, kemungkinan akibat ukuran kecil dan tumpang tindih antarobjek (*occlusion*).
- **Masih ada kesalahan klasifikasi**, terutama pada kelas dengan data sedikit seperti ambulans.
- Hasil pada foto lalu lintas padat sebaiknya dibaca sebagai bukti bahwa model bisa bekerja pada gambar baru, **bukan** bukti bahwa model siap dipakai untuk menghitung seluruh kendaraan di jalan raya.

---

## 🚀 Cara Menjalankan

1. Buka notebook di **Google Colab** dan pilih runtime **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
2. Instal pustaka pendukung:
   ```bash
   pip install roboflow pycocotools -q
   ```
3. Simpan API key Roboflow di **Colab Secrets** dengan nama `ROBOFLOW_API_KEY`. Jangan menulis API key langsung di dalam kode, terutama jika notebook dipublikasikan.
4. Jalankan seluruh sel secara berurutan. Notebook akan mengunduh dataset, melatih model, mengevaluasi, lalu menyimpan bobot terbaik ke `rcnn_best_model.pth`.
5. Untuk mencoba model pada gambar sendiri, ubah `url_test` pada bagian uji coba menjadi URL gambar atau path berkas lokal:
   ```python
   predict_single_image(best_model, "URL_ATAU_PATH_GAMBAR", score_thr=0.5)
   ```

> 💡 Menurunkan `score_thr` (misalnya 0.3) memunculkan lebih banyak deteksi, termasuk objek jauh, dengan risiko lebih banyak salah deteksi.

---

## 🔮 Rencana Pengembangan

- Melatih dengan seluruh data latih dan jumlah epoch lebih banyak
- Mencoba `fasterrcnn_mobilenet_v3_large_fpn` (resolusi input lebih besar) atau backbone ResNet-50 untuk meningkatkan deteksi objek kecil
- Menambahkan *data augmentation* (flip horizontal, perubahan kecerahan)
- Menambah data untuk kelas yang jarang muncul (misalnya ambulans) dan memeriksa *confusion matrix* per kelas

---

## 📝 Artikel Lengkap

Penjelasan lengkap tiap bagian kode tersedia di artikel Medium:
**[Mendeteksi Kendaraan dengan Faster R-CNN MobileNetV3: Dari Dataset Roboflow hingga Uji Coba di Jalan Raya](https://medium.com/@m.alfarisy797/mendeteksi-kendaraan-dengan-faster-r-cnn-mobilenetv3-dari-dataset-roboflow-hingga-uji-coba-di-cd7b7ae166bf?sharedUserId=m.alfarisy797)** <!-- ganti # dengan link artikel Medium Anda -->

---

## 👤 Penulis

**Muhammad Al Farisy**
UNPAM MTI REG C, Computer Vision Advance

---

## 🙏 Referensi

- [Torchvision Detection Models](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- [pycocotools](https://github.com/cocodataset/cocoapi)
- [Roboflow](https://roboflow.com/)
