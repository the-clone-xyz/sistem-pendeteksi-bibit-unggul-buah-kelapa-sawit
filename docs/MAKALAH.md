# Makalah Sistem Pendeteksi Bibit Unggul Buah Kelapa Sawit Berbasis Citra

## Identitas Makalah

**Judul:** Sistem Pendeteksi Bibit Unggul Buah Kelapa Sawit Berbasis Citra Menggunakan Deep Learning dan Parameter Visual

**Nama Penulis:** [Isi nama penulis]

**Program Studi:** [Isi program studi]

**Institusi:** [Isi nama kampus/sekolah]

**Tahun:** 2026

---

## Abstrak

Penelitian ini membahas pengembangan sistem pendeteksi bibit unggul buah kelapa sawit berbasis citra. Sistem menerima unggahan satu atau beberapa gambar sawit, melakukan klasifikasi tingkat kematangan menggunakan model deep learning, mengekstraksi parameter visual berupa ukuran gambar, estimasi area objek, dan warna dominan, kemudian menentukan status bibit unggul berdasarkan parameter yang tersimpan pada database SQLite. Kelas yang digunakan meliputi `matang`, `setengah_matang`, dan `mentah`. Hasil sistem ditampilkan melalui antarmuka Streamlit dalam bentuk ringkasan pencarian, detail prediksi, skor kelas, status unggul, dan riwayat prediksi. Sistem ini diharapkan membantu proses seleksi awal kandidat bibit unggul secara lebih objektif dibandingkan pengamatan manual.

**Kata kunci:** kelapa sawit, bibit unggul, klasifikasi citra, deep learning, tingkat kematangan, warna dominan, SQLite, Streamlit.

---

## Format Peletakan File Gambar

Letakkan file gambar pendukung makalah pada folder berikut:

```text
docs/
  MAKALAH.md
  gambar/
    gambar-01-arsitektur-sistem.png
    gambar-02-alur-prediksi.png
    gambar-03-tampilan-upload.png
    gambar-04-ringkasan-pencarian.png
    gambar-05-detail-parameter-visual.png
    gambar-06-skema-database.png
    gambar-07-confusion-matrix.png
    gambar-08-contoh-dataset.png
```

Format penulisan gambar di dalam makalah:

```markdown
![Gambar 1. Arsitektur sistem](gambar/gambar-01-arsitektur-sistem.png)

Gambar 1. Arsitektur sistem pendeteksi bibit unggul sawit.
```

Aturan penamaan gambar:

- Gunakan huruf kecil.
- Gunakan tanda hubung `-`, bukan spasi.
- Awali dengan nomor urut, contoh `gambar-03-tampilan-upload.png`.
- Simpan gambar relatif terhadap file ini di folder `docs/gambar/`.
- Jangan memakai link gambar dari internet untuk gambar hasil sistem; gunakan screenshot atau diagram milik sendiri.

Daftar gambar yang disarankan:

| No. | Nama File | Isi Gambar | Diletakkan Pada Bagian |
| --- | --- | --- | --- |
| 1 | `gambar-01-arsitektur-sistem.png` | Diagram komponen sistem: user, Streamlit, model, ekstraksi visual, SQLite | Bab III |
| 2 | `gambar-02-alur-prediksi.png` | Flowchart upload gambar sampai keputusan unggul | Bab III |
| 3 | `gambar-03-tampilan-upload.png` | Screenshot form upload gambar | Bab IV |
| 4 | `gambar-04-ringkasan-pencarian.png` | Screenshot ringkasan Bibit Unggul/Belum Unggul | Bab IV |
| 5 | `gambar-05-detail-parameter-visual.png` | Screenshot ukuran, area objek, kematangan, warna dominan | Bab IV |
| 6 | `gambar-06-skema-database.png` | Skema tabel `predictions` dan `superior_seed_parameters` | Bab III |
| 7 | `gambar-07-confusion-matrix.png` | Confusion matrix hasil evaluasi model | Bab IV |
| 8 | `gambar-08-contoh-dataset.png` | Contoh gambar kelas matang, setengah matang, mentah | Bab III |

---

# BAB I PENDAHULUAN

## 1.1 Latar Belakang

Kelapa sawit merupakan komoditas penting yang membutuhkan proses seleksi dan penilaian kualitas buah secara tepat. Dalam praktik lapangan, penilaian tingkat kematangan dan kelayakan buah sering dilakukan secara visual oleh manusia. Cara manual memiliki kelemahan karena dipengaruhi pengalaman pengamat, pencahayaan, jarak pengambilan gambar, dan variasi warna buah. Beberapa penelitian terbaru menunjukkan bahwa computer vision dan deep learning dapat digunakan untuk mendeteksi atau mengklasifikasikan tingkat kematangan tandan buah segar kelapa sawit secara non-destruktif [1], [2], [3].

Program ini dikembangkan untuk membantu proses pencarian kandidat bibit unggul buah kelapa sawit berbasis citra. Sistem menerima file gambar dari user, memprediksi kelas kematangan, menganalisis parameter visual seperti ukuran, area objek, dan warna dominan, kemudian menyimpan hasilnya pada database SQLite. Dengan adanya sistem ini, proses seleksi awal dapat dilakukan lebih cepat, terdokumentasi, dan mudah ditelusuri kembali melalui riwayat prediksi.

## 1.2 Rumusan Masalah

1. Bagaimana membangun sistem berbasis citra untuk mendeteksi kandidat bibit unggul buah kelapa sawit?
2. Bagaimana sistem menilai tingkat kematangan, ukuran, dan warna dominan dari gambar yang diunggah?
3. Bagaimana hasil prediksi dan parameter bibit unggul dapat disimpan ke dalam database?
4. Bagaimana menampilkan hasil pencarian bibit unggul agar mudah dipahami user?

## 1.3 Batasan Masalah

1. Sistem hanya menerima file gambar dengan format `jpg`, `jpeg`, `png`, `bmp`, dan `webp`.
2. Kelas prediksi yang digunakan adalah `matang`, `setengah_matang`, dan `mentah`.
3. Ukuran yang dihitung adalah ukuran citra dan estimasi area objek dalam persen, bukan ukuran fisik dalam centimeter.
4. Warna dominan dihitung dari citra menggunakan pendekatan HSV sederhana.
5. Sistem tidak melakukan kalibrasi kamera atau pengukuran fisik lapangan.
6. Database yang digunakan adalah SQLite lokal.

## 1.4 Tujuan Penelitian

1. Membuat sistem upload gambar untuk mencari kandidat bibit unggul sawit.
2. Mengintegrasikan model klasifikasi citra untuk menentukan tingkat kematangan.
3. Menambahkan parameter visual berupa ukuran, estimasi area objek, dan warna dominan.
4. Menyimpan hasil prediksi, status unggul, skor parameter, dan snapshot parameter ke database.
5. Menyediakan tampilan hasil yang informatif melalui Streamlit.

## 1.5 Manfaat Penelitian

1. Membantu seleksi awal kandidat bibit unggul secara lebih objektif.
2. Mempermudah pencatatan hasil prediksi dan parameter visual.
3. Menyediakan dasar pengembangan sistem grading sawit berbasis citra.
4. Menjadi bahan pembelajaran penerapan deep learning pada bidang pertanian.

---

# BAB II TINJAUAN PUSTAKA

## 2.1 Kelapa Sawit dan Tingkat Kematangan Buah

Tingkat kematangan buah kelapa sawit berkaitan dengan perubahan warna dan kualitas minyak yang dihasilkan. Pada buah yang semakin matang, warna dapat berubah dari gelap/hijau menuju oranye, merah, atau coklat. Penilaian tingkat kematangan secara manual masih umum dilakukan, tetapi memiliki potensi subjektivitas dan inkonsistensi. Kajian sistematis oleh Lai et al. menunjukkan bahwa metode computer vision dan deep learning menjadi pendekatan yang relevan untuk klasifikasi kematangan tandan buah segar [1].

## 2.2 Computer Vision untuk Klasifikasi Kematangan Sawit

Computer vision memungkinkan sistem mengekstraksi informasi dari gambar, seperti warna, bentuk, tekstur, dan pola visual. Penelitian Mansour et al. membandingkan beberapa algoritma object detection untuk klasifikasi kematangan tandan buah segar sawit, termasuk MobileNetV2 SSD, EfficientDet, dan YOLOv5 [2]. Penelitian lain juga menunjukkan penggunaan model deep learning pada data gambar maupun video untuk mendukung deteksi kematangan secara real-time [3], [4].

## 2.3 Deep Learning dan Transfer Learning

Deep learning, khususnya Convolutional Neural Network (CNN), banyak digunakan untuk klasifikasi citra karena mampu mempelajari fitur visual secara otomatis. Suharjito et al. meneliti klasifikasi kematangan tandan buah segar pada perangkat mobile menggunakan beberapa arsitektur CNN ringan dan pendekatan transfer learning [5]. Program ini menggunakan pendekatan serupa, yaitu memanfaatkan model klasifikasi citra untuk membedakan kelas `matang`, `setengah_matang`, dan `mentah`.

## 2.4 Parameter Visual: Ukuran, Kematangan, dan Warna

Pada sistem ini, parameter visual terdiri dari:

1. **Ukuran gambar:** dimensi gambar dalam piksel.
2. **Area objek:** estimasi persentase area objek sawit terhadap gambar.
3. **Tingkat kematangan:** kelas hasil prediksi model.
4. **Warna dominan:** warna yang paling dominan pada area objek berdasarkan analisis HSV.

Parameter warna relevan karena beberapa penelitian menyebut warna buah sebagai indikator visual penting dalam penentuan kematangan sawit [1], [6]. Namun, parameter ukuran pada sistem ini masih berupa estimasi berbasis citra sehingga belum setara dengan pengukuran fisik.

## 2.5 Database SQLite untuk Riwayat Prediksi

SQLite digunakan untuk menyimpan riwayat prediksi dan parameter bibit unggul. Data yang disimpan meliputi nama file, hash gambar, kelas prediksi, confidence, rekomendasi, sumber prediksi, skor kelas, status unggul, skor parameter, serta snapshot parameter yang digunakan saat prediksi. Penyimpanan snapshot penting agar histori tetap dapat diaudit walaupun parameter unggul diubah kemudian.

---

# BAB III METODOLOGI PENELITIAN

## 3.1 Desain Sistem

Sistem terdiri dari beberapa komponen utama:

1. User mengunggah gambar melalui antarmuka Streamlit.
2. Sistem memuat gambar dan menghitung hash file.
3. Sistem mengekstraksi parameter visual dari gambar.
4. Model deep learning memprediksi tingkat kematangan.
5. Sistem menilai status unggul berdasarkan parameter pada database.
6. Hasil prediksi dan snapshot parameter disimpan ke SQLite.
7. User melihat ringkasan pencarian, detail parameter visual, dan riwayat prediksi.

Letakkan diagram arsitektur di bagian ini:

```markdown
![Gambar 1. Arsitektur sistem](gambar/gambar-01-arsitektur-sistem.png)

Gambar 1. Arsitektur sistem pendeteksi bibit unggul sawit.
```

## 3.2 Alur Sistem

Alur kerja sistem dapat ditulis sebagai berikut:

1. User memilih satu atau beberapa file gambar.
2. Sistem memvalidasi format file.
3. Gambar diproses menjadi input model.
4. Model menghasilkan probabilitas untuk setiap kelas.
5. Kelas dengan probabilitas tertinggi dipilih sebagai hasil prediksi.
6. Sistem menghitung parameter ukuran, area objek, dan warna dominan.
7. Parameter hasil prediksi dibandingkan dengan target parameter unggul.
8. Sistem memberi status `Unggul`, `Perlu Pemeriksaan Lanjutan`, atau `Tidak Unggul`.
9. Hasil disimpan ke database.
10. Hasil ditampilkan pada tabel ringkasan dan detail prediksi.

Letakkan flowchart di bagian ini:

```markdown
![Gambar 2. Alur prediksi sistem](gambar/gambar-02-alur-prediksi.png)

Gambar 2. Alur prediksi sistem dari upload gambar sampai status unggul.
```

## 3.3 Dataset

Dataset disusun ke dalam tiga split:

```text
dataset/
  train/
    matang/
    setengah_matang/
    mentah/
  validation/
    matang/
    setengah_matang/
    mentah/
  test/
    matang/
    setengah_matang/
    mentah/
```

Contoh gambar dataset dapat diletakkan pada bagian ini:

```markdown
![Gambar 8. Contoh dataset](gambar/gambar-08-contoh-dataset.png)

Gambar 8. Contoh gambar pada kelas matang, setengah matang, dan mentah.
```

## 3.4 Model Klasifikasi

Model klasifikasi citra digunakan untuk memprediksi tiga kelas:

| Kelas | Makna | Rekomendasi |
| --- | --- | --- |
| `matang` | Buah terdeteksi matang | Layak / Direkomendasikan |
| `setengah_matang` | Buah belum sepenuhnya matang | Perlu Pemeriksaan Lanjutan |
| `mentah` | Buah belum matang | Tidak Direkomendasikan |

Model yang digunakan dapat berupa CNN sederhana atau MobileNetV2. MobileNetV2 relevan untuk sistem klasifikasi citra ringan karena pernah digunakan dalam studi klasifikasi/deteksi kematangan sawit dan object detection berbasis MobileNetV2 SSD [2], [5].

## 3.5 Ekstraksi Parameter Visual

Parameter visual yang diekstraksi sistem adalah:

| Parameter | Sumber Nilai | Keterangan |
| --- | --- | --- |
| Ukuran gambar | Dimensi citra | Lebar dan tinggi gambar dalam piksel |
| Area objek | Masking sederhana berbasis saturasi dan value HSV | Estimasi persen area objek sawit |
| Tingkat kematangan | Output model klasifikasi | `matang`, `setengah_matang`, atau `mentah` |
| Warna dominan | Analisis HSV pada area objek | Merah, oranye, kuning, hijau, coklat, gelap |
| Confidence | Output probabilitas model | Keyakinan model terhadap kelas terpilih |

## 3.6 Parameter Bibit Unggul

Parameter default sistem:

| Parameter | Target Default | Bobot Default |
| --- | --- | --- |
| Tingkat kematangan | `matang` | 0.35 |
| Confidence minimum | 75% | 0.25 |
| Rekomendasi | Layak / Direkomendasikan | 0.10 |
| Ukuran area objek minimum | 20% | 0.15 |
| Warna dominan | merah, oranye, coklat | 0.10 |

Nilai parameter dapat diubah melalui sidebar aplikasi. Setiap hasil prediksi menyimpan snapshot parameter sehingga riwayat hasil tetap konsisten.

## 3.7 Skema Database

Tabel utama yang digunakan:

1. `predictions`
2. `superior_seed_parameters`

Skema ringkas:

```text
predictions
- id
- created_at
- filename
- image_sha256
- class_name
- display_label
- confidence
- recommendation
- source
- model_path
- probabilities_json
- quality_status
- quality_score
- parameter_snapshot_json

superior_seed_parameters
- id
- code
- label
- target_value
- unit
- weight
- description
- is_active
- created_at
- updated_at
```

Letakkan skema database di bagian ini:

```markdown
![Gambar 6. Skema database](gambar/gambar-06-skema-database.png)

Gambar 6. Skema database sistem pendeteksi bibit unggul sawit.
```

---

# BAB IV HASIL DAN PEMBAHASAN

## 4.1 Tampilan Upload Gambar

Sistem menyediakan fitur upload satu atau beberapa gambar. Setelah gambar diunggah, sistem langsung melakukan prediksi dan menampilkan hasil pencarian.

```markdown
![Gambar 3. Tampilan upload gambar](gambar/gambar-03-tampilan-upload.png)

Gambar 3. Tampilan upload gambar pada aplikasi Streamlit.
```

## 4.2 Ringkasan Pencarian Bibit Unggul

Ringkasan pencarian menampilkan jumlah file yang diuji, jumlah kandidat bibit unggul, dan jumlah file yang belum memenuhi kriteria unggul.

```markdown
![Gambar 4. Ringkasan pencarian](gambar/gambar-04-ringkasan-pencarian.png)

Gambar 4. Ringkasan hasil pencarian bibit unggul.
```

## 4.3 Detail Parameter Visual

Pada setiap file, sistem menampilkan ukuran gambar, area objek, tingkat kematangan, warna dominan, confidence, status unggul, dan skor parameter.

```markdown
![Gambar 5. Detail parameter visual](gambar/gambar-05-detail-parameter-visual.png)

Gambar 5. Detail parameter visual hasil analisis gambar.
```

## 4.4 Evaluasi Model

Hasil evaluasi model utama `models/model_sawit.keras` pada split `dataset/test`:

| Metrik | Nilai |
| --- | --- |
| Total data test | 135 gambar |
| Prediksi benar | 107 gambar |
| Prediksi salah | 28 gambar |
| Akurasi test | 79,26% |

Akurasi per kelas:

| Kelas | Akurasi | Benar / Total |
| --- | --- | --- |
| `matang` | 84,44% | 38 / 45 |
| `setengah_matang` | 66,67% | 30 / 45 |
| `mentah` | 86,67% | 39 / 45 |

Confusion matrix dapat diletakkan pada bagian ini:

```markdown
![Gambar 7. Confusion matrix](gambar/gambar-07-confusion-matrix.png)

Gambar 7. Confusion matrix hasil evaluasi model.
```

## 4.5 Pembahasan

Berdasarkan hasil evaluasi, model mampu mengenali kelas `matang` dan `mentah` dengan akurasi lebih tinggi dibandingkan `setengah_matang`. Hal ini dapat terjadi karena kelas `setengah_matang` memiliki ciri visual yang berada di antara matang dan mentah, sehingga lebih mudah tertukar dengan kelas lain. Penambahan parameter visual seperti warna dominan dan area objek membantu memberikan konteks tambahan pada hasil prediksi, tetapi parameter tersebut masih bersifat estimasi karena belum menggunakan segmentasi objek yang presisi atau pengukuran fisik.

Sistem ini cocok digunakan sebagai alat bantu seleksi awal. Untuk penggunaan lapangan, sistem masih perlu ditingkatkan dengan dataset yang lebih besar, variasi pencahayaan, variasi sudut pengambilan gambar, validasi lapangan, serta metode segmentasi objek yang lebih akurat.

---

# BAB V KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

1. Sistem berhasil menyediakan fitur upload gambar untuk mencari kandidat bibit unggul buah kelapa sawit.
2. Sistem dapat memprediksi tingkat kematangan menggunakan model deep learning.
3. Sistem dapat menampilkan parameter visual berupa ukuran gambar, estimasi area objek, tingkat kematangan, dan warna dominan.
4. Sistem menyimpan riwayat prediksi dan parameter unggul ke database SQLite.
5. Berdasarkan evaluasi awal, model utama memperoleh akurasi test sebesar 79,26% pada data test yang tersedia.

## 5.2 Saran

1. Menambah jumlah dataset dan variasi kondisi pengambilan gambar.
2. Menggunakan metode segmentasi objek agar estimasi area objek lebih akurat.
3. Menambahkan kalibrasi kamera atau referensi ukuran jika ingin mengukur ukuran fisik dalam centimeter.
4. Menguji model pada data lapangan yang belum pernah dilihat model.
5. Menambahkan laporan otomatis, misalnya export hasil prediksi ke CSV atau PDF.

---

# Daftar Pustaka

Catatan: Referensi berikut dipilih dari publikasi nyata dalam rentang 2021-2026 dan relevan dengan klasifikasi kematangan sawit, computer vision, deep learning, dataset sawit, dan sistem pendeteksi berbasis citra.

[1] Lai, J. W., Ramli, H. R., Ismail, L. I., & Wan Hasan, W. Z. (2023). Oil Palm Fresh Fruit Bunch Ripeness Detection Methods: A Systematic Review. *Agriculture, 13*(1), 156. https://doi.org/10.3390/agriculture13010156

[2] Mansour, M. Y. M. A., Dambul, K. D., & Choo, K. Y. (2022). Object Detection Algorithms for Ripeness Classification of Oil Palm Fresh Fruit Bunch. *International Journal of Technology, 13*(6), 1326-1335. https://doi.org/10.14716/ijtech.v13i6.5932

[3] Junior, F. A., & Suharjito. (2023). Video based oil palm ripeness detection model using deep learning. *Heliyon, 9*(1), e13036. https://doi.org/10.1016/j.heliyon.2023.e13036

[4] Suharjito, Junior, F. A., Koeswandy, Y. P., Debi, Nurhayati, P. W., Asrol, M., & Marimin. (2023). Annotated Datasets of Oil Palm Fruit Bunch Piles for Ripeness Grading Using Deep Learning. *Scientific Data, 10*, 72. https://doi.org/10.1038/s41597-023-01958-x

[5] Suharjito, Elwirehardja, G. N., & Prayoga, J. S. (2021). Oil palm fresh fruit bunch ripeness classification on mobile devices using deep learning approaches. *Computers and Electronics in Agriculture, 188*, 106359. https://doi.org/10.1016/j.compag.2021.106359

[6] Goh, J. Y., Yunos, Y. M., & Mohamed Ali, M. S. (2024). Fresh Fruit Bunch Ripeness Classification Methods: A Review. *Food and Bioprocess Technology*. Published online 28 June 2024. https://doi.org/10.1007/s11947-024-03483-0

[7] Bonet, I., Gongora, M., Acevedo, F., & Ochoa, I. (2024). Deep Learning Model to Predict the Ripeness of Oil Palm Fruit. In *Proceedings of the 16th International Conference on Agents and Artificial Intelligence (ICAART 2024)*, 1068-1075. https://doi.org/10.5220/0012434600003636

[8] Puttinaovarat, S., Horkaew, P., Jandang, C., & Kongcharoen, M. (2024). Oil Palm Bunch Ripeness Classification and Plantation Verification Platform: Leveraging Deep Learning and Geospatial Analysis and Visualization. *ISPRS International Journal of Geo-Information, 13*(5), 158. https://doi.org/10.3390/ijgi13050158

---

# Lampiran A. Perintah yang Dapat Dicantumkan

Menjalankan aplikasi:

```bash
streamlit run app.py
```

Evaluasi model pada data test:

```bash
python3 scripts/evaluate_model.py --dataset dataset --model models/model_sawit.keras --split test
```

Training model:

```bash
python3 src/train.py
```

---

# Lampiran B. Checklist Penyusunan Makalah

- [ ] Lengkapi identitas penulis.
- [ ] Masukkan screenshot tampilan aplikasi ke `docs/gambar/`.
- [ ] Masukkan diagram arsitektur sistem.
- [ ] Masukkan flowchart alur prediksi.
- [ ] Masukkan confusion matrix hasil evaluasi.
- [ ] Pastikan semua gambar memiliki nomor dan keterangan.
- [ ] Pastikan semua sitasi di isi makalah muncul di daftar pustaka.
- [ ] Pastikan referensi yang digunakan berasal dari 2021-2026.
