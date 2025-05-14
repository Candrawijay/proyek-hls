import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Cek apakah CUDA tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan device: {device}")

# Load model multilingual
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

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

# Encode SDG labels
sdg_texts = list(sdg_labels.values())
sdg_embeddings = model.encode(sdg_texts, convert_to_tensor=True)

# Baca file Excel
input_path = "preprocessed_output.csv"
output_path = "labeled.csv"

df = pd.read_excel(input_path)

# Pastikan kolom 'processed_text' ada
if 'processed_text' not in df.columns:
    raise ValueError("Kolom 'processed_text' tidak ditemukan di file Excel!")

# Fungsi klasifikasi berdasarkan similarity
def classify(text):
    if pd.isna(text) or len(str(text).strip()) == 0:
        return "Tidak diklasifikasikan"
    query_embedding = model.encode(str(text), convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, sdg_embeddings)[0]
    best_match_idx = int(cosine_scores.argmax())
    return sdg_labels[best_match_idx + 1]

# Proses klasifikasi
print("Memulai pelabelan otomatis...")
df['Kategori SDGs'] = df['processed_text'].apply(classify)

# Simpan hasil ke Excel
df.to_excel(output_path, index=False)
print(f"Pelabelan selesai. Hasil disimpan sebagai: {output_path}")
