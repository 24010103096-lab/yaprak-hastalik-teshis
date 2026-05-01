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

# --- 2. TAHMİN FONKSİYONU (HATA DÜZELTİCİ) ---
def predict_image(image_data, model):
    # Modelin giriş boyutunu otomatik öğren (Hata almamak için kritik)
    input_shape = model.input_shape[1:3] # Genelde (224, 224)
    
    # Resmi modelin istediği boyuta getir
    img = image_data.convert('RGB').resize(input_shape)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    
    preds = model.predict(arr)
    return preds

# --- 3. ANA ARAYÜZ ---
st.title("🌿 Akıllı Tarım & Bitki Sağlık İstasyonu")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Görüntü Analizi")
    src = st.radio("Kaynak:", ["Kamera", "Dosya"], horizontal=True)
    img_input = st.camera_input("Fotoğraf") if src == "Kamera" else st.file_uploader("Yükle", type=["jpg", "png", "jpeg"])

with col2:
    if img_input:
        image = Image.open(img_input)
        st.image(image, use_container_width=True)
        
        if st.button("🔎 ANALİZİ BAŞLAT"):
            with st.spinner("İnceleniyor..."):
                try:
                    preds = predict_image(image, model)
                    res_idx = str(np.argmax(preds[0]))
                    res_name = class_names[res_idx]
                    conf = np.max(preds[0]) * 100
                    
                    st.success(f"**Teşhis:** {res_name}")
                    st.metric("Güven Oranı", f"%{conf:.2f}")
                except Exception as e:
                    st.error(f"Analiz sırasında bir hata oluştu: {e}")
