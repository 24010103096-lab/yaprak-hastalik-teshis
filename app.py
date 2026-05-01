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

# --- 2. TEDAVİ REHBERİ (Geliştirmelerin Geri Geldi) ---
REHBER = {
    "Apple___Apple_scab": "🍎 **Elma Karalekesi:** Dökülen yaprakları temizleyin. Erken dönemde fungisit uygulayın.",
    "Potato___Late_blight": "🥔 **Patates Geç Yanıklığı:** Çok tehlikelidir! Nem kontrolü sağlayın ve acil ilaçlama yapın.",
    "Tomato_Late_blight": "🍅 **Domates Geç Yanıklığı:** Hastalıklı yaprakları imha edin. Bitkiyi stresten koruyun.",
    "healthy": "✅ **Sağlıklı:** Bitkiniz gayet iyi durumda! Bakıma devam.",
    "default": "⚠️ **Öneri:** Bitkiyi izole edin ve bir ziraat uzmanına numune gösterin."
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
st.title("🌿 Akıllı Bitki Sağlık İstasyonu")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Analiz Paneli")
    src = st.radio("Giriş:", ["Kamera", "Dosya"], horizontal=True)
    img_in = st.camera_input("Fotoğraf") if src == "Kamera" else st.file_uploader("Yükle", type=["jpg","png","jpeg"])
    
    st.divider()
    st.subheader("📜 Teşhis Geçmişi")
    if os.path.exists("teshis_gecmisi.csv"):
        st.table(pd.read_csv("teshis_gecmisi.csv").tail(5))

with col2:
    if img_in:
        image = Image.open(img_in)
        st.image(image, use_container_width=True)
        
        if st.button("🔎 ANALİZİ BAŞLAT"):
            try:
                # Dinamik Boyutlandırma (Hata Almamak İçin)
                target_size = model.input_shape[1:3]
                img = image.convert('RGB').resize(target_size)
                arr = np.array(img) / 255.0
                arr = np.expand_dims(arr, axis=0).astype(np.float32)
                
                # Tahmin
                preds = model.predict(arr)
                idx = np.argmax(preds[0])
                conf = np.max(preds[0]) # 0 ile 1 arasında değer
                
                # Güven oranı düzeltmesi (Hatalı %2000 rakamını önler)
                if conf > 1.0: conf = 1.0 
                display_conf = conf * 100
                
                res_name = class_names[str(idx)]
                
                # Kaydet ve Göster
                save_log(res_name, f"%{display_conf:.2f}")
                st.success(f"**Teşhis:** {res_name}")
                st.metric("Güven Oranı", f"%{display_conf:.2f}")
                
                # Tedavi Önerisi
                tavsiye = next((v for k, v in REHBER.items() if k in res_name), REHBER["default"])
                st.warning(tavsiye)
                
            except Exception as e:
                st.error(f"Hata oluştu: {e}")
