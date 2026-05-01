import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime
import pandas as pd

# --- SAYFA AYARLARI (Mobil Uyumluluk İçin) ---
st.set_page_config(page_title="Bitki Doktoru", page_icon="🌿", layout="wide")

# Özel Tasarım (Mobil butonlar ve arka plan)
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #2e7d32; color: white; padding: 10px; font-size: 18px; font-weight: bold;}
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

# --- 2. TEDAVİ REHBERİ ---
REHBER = {
    "Apple___Apple_scab": "🍎 **Elma Karalekesi:** Dökülen yaprakları temizleyin. Erken dönemde fungisit uygulayın.",
    "Corn_(maize)___Common_rust_": "🌽 **Mısır Pası:** Nem kontrolü yapın, hava sirkülasyonunu artırın.",
    "Potato___Late_blight": "🥔 **Patates Mildiyösü:** Acil müdahale gerekir! Bakırlı ilaçlar kullanın.",
    "Tomato_Late_blight": "🍅 **Domates Geç Yanıklığı:** Hastalıklı yaprakları imha edin.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "🍅 **Sarı Yaprak Kıvırcıklığı:** Beyaz sinekle mücadele edin.",
    "healthy": "✅ **Sağlıklı:** Bitkiniz gayet iyi durumda! Bakıma devam.",
    "default": "⚠️ **Öneri:** Bitkiyi izole edin ve bir ziraat uzmanına danışın."
}

# --- 3. GEÇMİŞ KAYIT FONKSİYONU ---
def save_log(result, confidence):
    log_file = "teshis_gecmisi.csv"
    now = datetime.now().strftime("%d-%m-%Y %H:%M")
    new_data = pd.DataFrame([[now, result, confidence]], columns=["Tarih", "Hastalık", "Güven"])
    if os.path.exists(log_file):
        new_data.to_csv(log_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(log_file, index=False)

# --- 4. ANA ARAYÜZ ---
st.title("🌿 Akıllı Tarım İstasyonu")

# Sol Panel / Telefon için Üst Panel
with st.sidebar:
    st.header("🌦️ Tarla Durumu")
    st.metric(label="Sıcaklık", value="24°C", delta="Nem: %82", delta_color="inverse")
    st.warning("⚠️ Nem yüksek. Mantar hastalıklarına dikkat!")
    
    st.divider()
    st.header("📜 Son Teşhisler")
    if os.path.exists("teshis_gecmisi.csv"):
        df = pd.read_csv("teshis_gecmisi.csv").tail(5)
        st.dataframe(df, use_container_width=True)

# Sağ Panel / Telefon için Ana İçerik
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Görüntü Analizi")
    src = st.radio("İşlem Seçin:", ["Kamera (Fotoğraf Çek)", "Galeriden Yükle"], horizontal=True)
    img_input = st.camera_input("Kamerayı Aç") if "Kamera" in src else st.file_uploader("Dosya Seç", type=["jpg", "png", "jpeg"])

with col2:
    if img_input:
        image = Image.open(img_input)
        st.image(image, use_container_width=True, caption="İncelenen Yaprak")
        
        if st.button("🔎 YAPAY ZEKAYI ÇALIŞTIR"):
            with st.spinner("Hücresel düzeyde analiz yapılıyor..."):
                try:
                    # 1. Hata Çözümü: Dinamik Boyutlandırma
                    target_size = model.input_shape[1:3]
                    img = image.convert('RGB').resize(target_size)
                    
                    arr = np.array(img) / 255.0
                    arr = np.expand_dims(arr, axis=0).astype(np.float32)
                    
                    # 2. Hata Çözümü: Softmax ile Gerçek Yüzde Oranı Bulma
                    preds = model.predict(arr)
                    probabilities = tf.nn.softmax(preds[0]).numpy() 
                    
                    idx = np.argmax(probabilities)
                    conf = np.max(probabilities) * 100
                    res_name = class_names[str(idx)]
                    
                    # Kayıt Sistemi
                    save_log(res_name, f"%{conf:.1f}")
                    
                    # Sonuçları Göster
                    st.success(f"**Teşhis Edildi:** {res_name}")
                    st.metric("Yapay Zeka Güven Skoru", f"%{conf:.1f}")
                    st.progress(int(conf))
                    
                    # Tedavi Önerisi
                    tavsiye = next((v for k, v in REHBER.items() if k in res_name), REHBER["default"])
                    st.info(tavsiye)
                    
                except Exception as e:
                    st.error(f"Model analiz hatası: {e}")
