import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Bitki Doktoru Pro", page_icon="🌿", layout="wide")
st.title("🌿 Akıllı Bitki Hastalık Teşhis & Tedavi Merkezi")
st.markdown("---")

# --- 1. MODEL VE ETİKETLERİ YÜKLEME ---
@st.cache_resource
def load_resources():
    try:
        model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if not model_files:
            return None, None
        model = tf.keras.models.load_model(model_files[0])
        with open('sinif_isimleri.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)
        return model, labels
    except:
        return None, None

model, class_names = load_resources()

# --- 2. GENİŞLETİLMİŞ TEDAVİ REHBERİ ---
REHBER = {
    "Apple___Apple_scab": "🍎 **Elma Karalekesi:** Dökülen yaprakları yok edin. Erken ilkbaharda fungisit uygulayın.",
    "Apple___Black_rot": "🍎 **Elma Siyah Çürüklüğü:** Ölü dalları budayın. Yara yerlerini aşı macunuyla kapatın.",
    "Apple___Cedar_apple_rust": "🍎 **Elma Pası:** Yakındaki ardıç ağaçlarını kontrol edin. Bakırlı ilaçlar kullanın.",
    "Corn_(maize)___Common_rust_": "🌽 **Mısır Pası:** Dayanıklı tohum seçin. Bitki sıklığını azaltarak hava akımını artırın.",
    "Corn_(maize)___Northern_Leaf_Blight": "🌽 **Kuzey Yaprak Yanıklığı:** Münavebe (ekim nöbeti) yapın. Hasat sonrası anızı gömün.",
    "Potato___Early_blight": "🥔 **Patates Erken Yanıklığı:** Potasyumlu gübreleme yapın. Bitkiyi strese sokacak susuzluktan kaçının.",
    "Potato___Late_blight": "🥔 **Patates Geç Yanıklığı (Mildiyö):** Çok tehlikelidir! Acilen uzman kontrolünde ilaçlama yapın.",
    "Tomato_Bacterial_spot": "🍅 **Domates Bakteriyel Leke:** Bakırlı preparatlar kullanın. Damlama sulama tercih edin.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "🍅 **Sarı Yaprak Kıvırcıklığı:** Beyaz sinek tuzakları kurun. Hastalıklı bitkiyi sökün.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "🍅 **Kırmızı Örümcek:** Nemi artırın. Akarisit içerikli ilaçlar kullanın.",
    "healthy": "✅ **Sağlıklı:** Bitkiniz sağlıklı görünüyor! Düzenli bakıma devam edin."
}

# --- 3. ANA ARAYÜZ ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Görüntü Kaynağı")
    tab1, tab2 = st.tabs(["Kamera", "Dosya"])
    with tab1: cam = st.camera_input("Canlı Fotoğraf")
    with tab2: up = st.file_uploader("Fotoğraf Yükle", type=["jpg", "png", "jpeg"])
    source = cam if cam else up

with col2:
    st.subheader("📊 Analiz Sonuçları")
    if source:
        img = Image.open(source)
        st.image(img, use_container_width=True)
        
        if st.button("Sistemi Çalıştır"):
            with st.spinner("Teşhis konuluyor..."):
                # Ön İşleme
                process_img = img.convert('RGB').resize((224, 224))
                arr = np.array(process_img) / 255.0
                arr = np.expand_dims(arr, axis=0)
                
                # Tahmin
                preds = model.predict(arr)
                res_idx = str(np.argmax(preds[0]))
                result = class_names[res_idx]
                conf = np.max(preds[0]) * 100
                
                # Çıktı
                st.success(f"**Teşhis:** {result}")
                st.metric("Doğruluk Oranı", f"%{conf:.2f}")
                
                # Tedavi Bulma
                key = next((k for k in REHBER if k in result), "default")
                st.info(REHBER.get(key, "⚠️ Bu durum için bir uzman görüşü almanız önerilir."))
