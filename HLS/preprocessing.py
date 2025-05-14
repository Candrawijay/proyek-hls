# Install library yang dibutuhkan (jalankan jika belum install)
# !pip install pandas nltk sastrawi scikit-learn langdetect googletrans==3.1.0a0

import pandas as pd
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from googletrans import Translator

# Download resource NLTK
nltk.download('stopwords')

# Baca file CSV
file_path = 'scraped_data.csv'

if not os.path.exists(file_path):
    print(f"File tidak ditemukan di: {file_path}")
    print(f"Direktori kerja saat ini: {os.getcwd()}")
    raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

df = pd.read_csv(file_path)

# Cek kolom teks yang akan diproses
text_column = 'Title'
if text_column not in df.columns:
    print("Kolom tidak ditemukan. Kolom-kolom yang tersedia:", df.columns)
    raise Exception(f"Kolom '{text_column}' tidak ada!")

# Inisialisasi stemmer, stopwords dan translator
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
translator = Translator()

# Fungsi preprocessing
def preprocess(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    try:
        if len(text.strip()) > 10:
            lang = detect(text)
            if lang == 'en':
                translated = translator.translate(text, src='en', dest='id')
                text = translated.text.lower()
    except:
        pass

    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Terapkan preprocessing
print(f"Memulai preprocessing kolom '{text_column}'...")
df['processed_text'] = df[text_column].apply(preprocess)
print("Preprocessing selesai!")

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

# Tampilkan hasil
print("\nContoh hasil preprocessing:")
print(df[[text_column, 'processed_text']].head())

print("\nBentuk matriks TF-IDF:", tfidf_matrix.shape)

# Simpan hasil
output_path = 'preprocessed_output.csv'
df.to_csv(output_path, index=False)
print(f"File hasil preprocessing disimpan sebagai {output_path}")
