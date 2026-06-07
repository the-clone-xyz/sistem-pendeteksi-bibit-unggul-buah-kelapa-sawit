# Sistem Pendeteksi Bibit Unggul Buah Kelapa Sawit

Aplikasi klasifikasi citra sawit untuk tiga kelas:

- `matang`
- `setengah_matang`
- `mentah`

Project ini bisa dijalankan sebagai GUI browser dengan Streamlit. User dapat upload satu atau beberapa file gambar untuk mencari kandidat bibit unggul berdasarkan hasil prediksi dan parameter unggul yang tersimpan di database. Sistem mendukung model validasi sawit/bukan-sawit melalui file `models/model_validasi_sawit.keras`; jika gambar dinilai bukan sawit, aplikasi menampilkan status `Yang kamu upload bukan gambar buah sawit`. Tampilan hasil menampilkan parameter ukuran gambar/area objek, tingkat kematangan, warna dominan, confidence, status unggul, dan rekomendasi. Jika model validasi belum tersedia, aplikasi memakai confidence model klasifikasi sebagai validasi sementara. Jika model klasifikasi belum tersedia, aplikasi memakai estimasi sementara berbasis warna. Setelah model dilatih, aplikasi otomatis memakai file model dari folder `models`.

## Database SQLite

Aplikasi memakai SQLite untuk menyimpan riwayat prediksi lokal. Database dibuat otomatis saat aplikasi dijalankan:

```text
data/predictions.sqlite3
```

Tabel utama:

```text
predictions
superior_seed_parameters
```

`predictions` menyimpan nama file, hash gambar, kelas prediksi, confidence, rekomendasi, sumber prediksi, path model, skor tiap kelas dalam format JSON, status unggul, skor parameter, dan snapshot parameter yang dipakai saat prediksi.

`superior_seed_parameters` menyimpan parameter bibit unggul yang dipakai sistem, seperti tingkat kematangan target `matang`, confidence minimum, ukuran area objek minimum, warna dominan target, rekomendasi target, dan bobot masing-masing parameter. Nilai parameter dapat diubah dari sidebar aplikasi dan otomatis tersimpan di SQLite.

## Struktur Dataset

Isi gambar sesuai struktur berikut:

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

Minimal folder `train` dan `validation` harus berisi gambar sebelum training.

## Menjalankan di GitHub Codespaces

1. Buka repository di GitHub.
2. Pilih `Code` -> `Codespaces` -> `Create codespace`.
3. Tunggu dependency selesai dipasang dari `requirements-training.txt`.
4. Upload dataset ke folder `dataset`.
5. Jalankan training:

```bash
python src/train.py
```

Model akan disimpan ke:

```text
models/model_sawit.keras
```

Model validasi sawit/bukan-sawit dapat diletakkan di:

```text
models/model_validasi_sawit.keras
```

Format output model validasi yang didukung:

- 1 output sigmoid: nilai mendekati 1 berarti `sawit`.
- 2 output softmax: urutan kelas `bukan_sawit`, `sawit`.

Jalankan aplikasi:

```bash
streamlit run app.py
```

Codespaces akan menampilkan port `8501`. Buka port tersebut untuk memakai aplikasi.

## Training Opsional

Training dengan epoch berbeda:

```bash
python src/train.py --epochs 30 --batch-size 16
```

Training dengan MobileNetV2:

```bash
python src/train.py --architecture mobilenetv2 --weights imagenet
```

Catatan: file dataset asli dan model hasil training diabaikan oleh `.gitignore` agar repository tetap ringan. Jika model perlu disimpan di GitHub, gunakan Git LFS atau hapus aturan ignore untuk file model yang dipilih.
File database SQLite di folder `data` juga diabaikan oleh `.gitignore` karena berisi data runtime lokal.

## Menjalankan Lokal

Install dependency GUI:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Install dependency lengkap untuk training:

```bash
pip install -r requirements-training.txt
python src/train.py
```
