# Sistem Pendeteksi Bibit Unggul Buah Kelapa Sawit

Aplikasi klasifikasi citra sawit untuk tiga kelas:

- `matang`
- `setengah_matang`
- `mentah`

Project ini bisa dijalankan sebagai GUI browser dengan Streamlit. Jika model belum tersedia, aplikasi memakai estimasi sementara berbasis warna. Setelah model dilatih, aplikasi otomatis memakai file model dari folder `models`.

## Database SQLite

Aplikasi memakai SQLite untuk menyimpan riwayat prediksi lokal. Database dibuat otomatis saat aplikasi dijalankan:

```text
data/predictions.sqlite3
```

Tabel utama:

```text
predictions
```

Data yang disimpan meliputi nama file, hash gambar, kelas prediksi, confidence, rekomendasi, sumber prediksi, path model, dan skor tiap kelas dalam format JSON.

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
