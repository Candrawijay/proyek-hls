import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Cek device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Menggunakan device: {device}")

# Load model yang ringan dan cepat
model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device=device)

# Daftar SDGs
sdg_labels = {
    1: "Tanpa Kemiskinan",
    2: "Tanpa Kelaparan",
    3: "Kesehatan dan Kesejahteraan",
    4: "Pendidikan Berkualitas",
    5: "Kesetaraan Gender",
    6: "Air Bersih dan Sanitasi Layak",
    7: "Energi Bersih dan Terjangkau",
    8: "Pekerjaan Layak dan Pertumbuhan Ekonomi",
    9: "Industri, Inovasi dan Infrastruktur",
    10: "Berkurangnya Kesenjangan",
    11: "Kota dan Permukiman Berkelanjutan",
    12: "Konsumsi dan Produksi Bertanggung Jawab",
    13: "Penanganan Perubahan Iklim",
    14: "Ekosistem Laut",
    15: "Ekosistem Darat",
    16: "Perdamaian, Keadilan dan Kelembagaan",
    17: "Kemitraan untuk Mencapai Tujuan"
}

# Encode label SDGs satu kali
sdg_texts = list(sdg_labels.values())
sdg_embeddings = model.encode(sdg_texts, convert_to_tensor=True)

# Baca file CSV
input_csv = "preprocessed_output.csv"
output_csv = "labeled_output.csv"

df = pd.read_csv(input_csv)

# Pastikan kolom 'processed_text' ada
if 'processed_text' not in df.columns:
    raise ValueError("Kolom 'processed_text' tidak ditemukan di file CSV!")

# Preprocessing teks
texts = df['processed_text'].fillna('').astype(str).tolist()
print(f"[INFO] Memulai proses pelabelan terhadap {len(texts)} baris...")

# Encode dalam batch
text_embeddings = model.encode(texts, convert_to_tensor=True, batch_size=64)

# Hitung cosine similarity
cosine_scores = util.cos_sim(text_embeddings, sdg_embeddings)

# Ambil index dengan skor tertinggi
predicted_indices = torch.argmax(cosine_scores, dim=1).cpu().numpy()
predicted_labels = [sdg_labels[i + 1] for i in predicted_indices]

# Tambahkan kolom hasil
df['Kategori SDGs'] = predicted_labels

# Simpan hasil sebagai CSV
df.to_csv(output_csv, index=False)
print(f"[INFO] Pelabelan selesai. File hasil disimpan sebagai '{output_csv}'.")
