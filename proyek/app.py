import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_and_save_model(csv_file):
    df = pd.read_csv(csv_file)

    # Gabungkan teks jika belum
    if "Combined_Text" not in df.columns:
        df["Combined_Text"] = df[["title", "ingredients", "steps"]].fillna('').agg(' '.join, axis=1)

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['ingredients'])

    # Simpan model dan data
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
    df.to_csv('data_clean.csv', index=False)
    print("✅ Model dan data berhasil disimpan.")

def load_model_and_data():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    df = pd.read_csv('data_clean.csv')
    return tfidf, tfidf_matrix, df

# Load model dan data
tfidf = joblib.load('https://github.com/trihadianto15/recommend_recipes/blob/main/proyek/tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('https://github.com/trihadianto15/recommend_recipes/blob/main/proyek/tfidf_matrix.pkl')
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
