import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Path aman relatif terhadap lokasi file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model dan data
tfidf_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
matrix_path = os.path.join(BASE_DIR, 'tfidf_matrix.pkl')
data_path = os.path.join(BASE_DIR, 'data_clean.csv')

try:
    tfidf = joblib.load(tfidf_path)
    tfidf_matrix = joblib.load(matrix_path)
    df = pd.read_csv(data_path)
except FileNotFoundError as e:
    st.error(f"Gagal memuat file: {e}")
    st.stop()

# Judul
st.title("🍲 Sistem Rekomendasi Resep Berdasarkan Bahan")
st.markdown("Masukkan bahan-bahan yang kamu punya, pisahkan dengan koma:")

# Input user
user_input = st.text_input("Contoh: ayam, bawang merah, cabai")

if st.button("Rekomendasikan"):
    if not user_input:
        st.warning("Silakan masukkan minimal satu bahan.")
    else:
        user_vec = tfidf.transform([user_input.lower()])
        scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:5]
        top_indices = top_indices.astype(int)  # pastikan indeks integer

        selected = df.loc[top_indices].copy()
        selected = selected[['Title', 'Ingredients']]
        selected['Similarity Score'] = scores[top_indices]
        recommendations = selected

        st.subheader("🍽️ Rekomendasi Resep:")
        for _, row in recommendations.iterrows():
            title = row['Title'] if pd.notna(row['Title']) else 'Tanpa Judul'
            ingredients = row['Ingredients'] if pd.notna(row['Ingredients']) else 'Tidak tersedia'

            st.markdown(f"#### {title}")
            st.markdown(f"**Bahan:** {ingredients}")
            st.markdown("---")
