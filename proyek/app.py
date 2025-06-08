import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load model dan data
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
df = pd.read_csv('data_clean.csv')

# Judul
st.title("🍽️ Sistem Rekomendasi Resep Berdasarkan Bahan")
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
        recommendations = df.iloc[top_indices][['Title', 'Ingredients']].copy()
        recommendations['Similarity Score'] = scores[top_indices]

        st.subheader("🍲 Rekomendasi Resep:")
        for i, row in recommendations.iterrows():
            st.markdown(f"### {row['Title']}")
            st.markdown(f"**Bahan:** {row['Ingredients']}")
            st.markdown("---")
