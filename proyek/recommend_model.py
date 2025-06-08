import joblib
import pandas as pd
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

