import streamlit as st
import pandas as pd
import re

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

# Load data dari file CSV
@st.cache_data
def load_data():
    df = pd.read_csv("dataclean_svm.csv")  # Path sesuai dengan file yang diupload
    return df

# Load dataset
try:
    data = load_data()
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

# Input dari pengguna
user_input = st.text_area("📝 Masukkan Komentar", height=150)

if st.button("🔍 Proses Komentar"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan komentar terlebih dahulu.")
    else:
        teks_bersih = preprocess(user_input)
        st.success("✅ Teks berhasil diproses.")
        st.markdown(f"<p>Hasil Preprocessing: <strong>{teks_bersih}</strong></p>", unsafe_allow_html=True)

# Tampilkan seluruh isi dataset
st.markdown("<h3>📄 Seluruh Dataset</h3>", unsafe_allow_html=True)
st.dataframe(data, use_container_width=True)

# Footer
st.markdown(""" 
<hr>
<div style='text-align: center;'>
    <small>© 2025 - Sistem Deteksi Komentar Ujaran Kebencian</small>
</div>
""", unsafe_allow_html=True)

