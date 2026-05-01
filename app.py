import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime
import pandas as pd

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Bitki Doktoru Pro", page_icon="🌿", layout="wide")

# --- CSS: Mobil Uyumluluk ve Tasarım ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #2e7d32; color: white; }
    .reportview-container .main .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. MODEL VE ETİKET YÜKLEME ---
@st.cache_resource
def load_resources():
    try:
        model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if not model_files: return None, None
        model = tf.keras.models.load_model(model_files[0])
        with open('sinif_isimleri.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)
        return model, labels
    except: return None, None

model, class_names = load_resources()

# --- 2. GELİŞMİŞ TEDAVİ VE BİLGİ BANKASI ---
REHBER = {
    "Apple___Apple_scab": "🍎 **Elma Karalekesi:** Enfekteli yaprakları temizleyin. Erken dönemde koruyucu fungisit kullanın.",
    "Corn_(maize)___Common_rust_": "🌽 **Mısır Pası:** Azotlu gübrelemeye dikkat edin. Hava sirkülasyonunu artırın.",
    "Potato___Late_blight": "🥔 **Patates Mildiyösü (Geç Yanıklık):** Acil müdahale gerekir! Bakırlı ilaçlar ve nem kontrolü şart.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "🍅 **Sarı Yaprak Kıvırcıklığı:** Gümüş renkli yansıtıcı malç kullanın. Beyaz sinekle mücadele edin.",
    "healthy": "✅ **Tebrikler:** Bitkiniz sağlıklı görünüyor. Mevcut bakım programına devam edin."
}

# --- 3. GEÇMİŞ KAYIT FONKSİYONU ---
def save_log(result, confidence):
    log_file = "teshis_gecmisi.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[now, result, confidence]], columns=["Tarih", "Hastalık", "Güven Skoru"])
    if os.path.exists(log_file):
        new_data.to_csv(log_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(log_file, index=False)

# --- 4. ANA ARAYÜZ ---
st.title("🌿 Akıllı Tarım & Bitki Sağlık İstasyonu")

# Sol Panel: Hava Durumu ve Bilgi
with st.sidebar:
    st.header("🌦️ Bölgesel Durum")
    # Örnek Hava Durumu Simülasyonu (Gerçek API anahtarı olmadan)
    st.metric("Sıcaklık", "24°C", "2°C")
    st.metric("Nem", "%82", "Yüksek Risk!")
    st.warning("⚠️ Yüksek nem sebebiyle mantar hastalığı riski artmıştır.")
    
    st.divider()
    st.header("📜 Son Teşhisler")
    if os.path.exists("teshis_gecmisi.csv"):
        df = pd.read_csv("teshis_gecmisi.csv").tail(5)
        st.table(df)

# Sağ Panel: Teşhis Ekranı
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Görüntü Analizi")
    src = st.radio("Seçiniz:", ["Kamera Kullan", "Dosya Yükle"], horizontal=True)
    img_input = st.camera_input("Fotoğraf") if src == "Kamera Kullan" else st.file_uploader("Yükle", type=["jpg", "png"])

with col2:
    if img_input:
        image = Image.open(img_input)
        st.image(image, use_container_width=True)
        
        if st.button("🔎 ANALİZİ BAŞLAT"):
            with st.spinner("Yapay Zeka İnceliyor..."):
                # İşleme
                img = image.convert('RGB').resize((224, 224))
                arr = np.array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)
                
                # Tahmin
                preds = model.predict(arr)
                res_idx = str(np.argmax(preds[0]))
                res_name = class_names[res_idx]
                conf = np.max(preds[0]) * 100
                
                # Kayıt
                save_log(res_name, f"%{conf:.2f}")
                
                # Sonuç
                st.success(f"**Sonuç:** {res_name}")
                st.progress(int(conf))
                
                # Tedavi Önerisi
                advice = next((v for k, v in REHBER.items() if k in res_name), "⚠️ Özel tedavi için ziraat mühendisine danışın.")
                st.info(advice)
