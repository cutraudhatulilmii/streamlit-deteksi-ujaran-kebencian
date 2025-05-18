import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Ujaran Kebencian", page_icon="💬", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 Deteksi Ujaran Kebencian</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan komentar untuk mengetahui apakah mengandung ujaran kebencian.</p>", unsafe_allow_html=True)

# Fungsi preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dataclean_svm.csv")
    df = df[['original_text', 'predicted_label_svm']]
    df.dropna(inplace=True)
    df['processed'] = df['original_text'].astype(str).apply(preprocess)
    return df

try:
    data = load_data()
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

# Split dan training ulang model
X = data['processed']
y = data['predicted_label_svm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = SVC(kernel='linear', probability=True)
model.fit(X_train_vec, y_train)

# Evaluasi model
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred, pos_label='ujaran kebencian') * 100

# Input user
user_input = st.text_area("📝 Masukkan Komentar", height=150)

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan komentar terlebih dahulu.")
    else:
        teks_bersih = preprocess(user_input)
        st.success("✅ Teks berhasil diproses.")
        st.markdown(f"<p>Hasil Preprocessing: <strong>{teks_bersih}</strong></p>", unsafe_allow_html=True)

        vector_input = vectorizer.transform([teks_bersih])
        hasil_prediksi = model.predict(vector_input)[0]

        if hasil_prediksi == "ujaran kebencian":
            st.error("🚨 Komentar ini mengandung *Ujaran Kebencian*!")
        else:
            st.success("👍 Komentar ini *tidak mengandung* ujaran kebencian.")

        # Tampilkan metrik
        st.markdown("<h4>📊 Evaluasi Model</h4>", unsafe_allow_html=True)
        st.markdown(f"<p>🎯 Akurasi Model: <strong>{acc:.2f}%</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p>🧠 F1-Score: <strong>{f1:.2f}%</strong></p>", unsafe_allow_html=True)

# Tampilkan data
st.markdown("<h3>📄 Seluruh Dataset</h3>", unsafe_allow_html=True)
st.dataframe(data[['original_text', 'predicted_label_svm']], use_container_width=True)

# Footer
st.markdown(""" 
<hr>
<div style='text-align: center;'>
    <small>© 2025 - Sistem Deteksi Komentar Ujaran Kebencian</small>
</div>
""", unsafe_allow_html=True)
