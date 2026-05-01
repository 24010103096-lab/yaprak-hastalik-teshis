import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Bitki Doktoru", page_icon="🌿", layout="centered")
st.title("🌿 Akıllı Bitki Hastalık Teşhis Sistemi")
st.markdown("---")

# --- 1. MODEL VE ETİKETLERİ YÜKLEME ---
@st.cache_resource
def load_resources():
    try:
        # Klasördeki .h5 dosyasını ismine bakmadan otomatik bulur
        model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if not model_files:
            st.error("Model dosyası (.h5) bulunamadı. Lütfen GitHub'a yüklediğinizden emin olun.")
            return None, None
        
        # Modeli yükle
        model = tf.keras.models.load_model(model_files[0])
        
        # Etiketleri (Sınıf İsimlerini) yükle
        with open('sinif_isimleri.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)
        return model, labels
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {e}")
        return None, None

model, class_names = load_resources()

# --- 2. TEDAVİ REHBERİ ---
TEDAVI_REHBERI = {
    "Tomato___Tomato_YellowLeaf__Curl_Virus": "🛑 **Hastalık:** Domates Sarı Yaprak Kıvırcıklığı Virüsü\n\n**Öneri:** Beyaz sineklerle mücadele edin. Hastalıklı bitkileri tarladan uzaklaştırın ve imha edin.",
    "Apple___Apple_scab": "🛑 **Hastalık:** Elma Karalekesi\n\n**Öneri:** Dökülen yaprakları temizleyin. Erken dönemde uygun fungisitlerle ilaçlama yapın.",
    "Corn_(maize)___Common_rust": "🛑 **Hastalık:** Mısır Pası\n\n**Öneri:** Dayanıklı tohum seçin. Bitki sıklığını azaltarak hava akımını artırın.",
    "default": "⚠️ **Öneri:** Bitkinin hastalıklı kısımlarını budayın. Kesin çözüm için bir ziraat uzmanına numune gösterin."
}

# --- 3. KULLANICI ARAYÜZÜ ---
if model is not None:
    tab1, tab2 = st.tabs(["📸 Fotoğraf Çek", "📁 Dosya Yükle"])

    with tab1:
        cam_file = st.camera_input("Yaprağın canlı fotoğrafını çekin")
    with tab2:
        uploaded_file = st.file_uploader("Veya galeriden fotoğraf seçin", type=["jpg", "jpeg", "png"])

    source = cam_file if cam_file else uploaded_file

    if source:
        image = Image.open(source)
        st.image(image, caption="Analiz Edilen Görüntü", use_container_width=True)
        
        if st.button("Teşhisi Başlat"):
            with st.spinner("Yapay Zeka Analiz Ediyor..."):
                try:
                    # Görüntü Ön İşleme (RGBA -> RGB çevrimi dahil)
                    img = image.convert('RGB').resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Tahmin Yapma
                    preds = model.predict(img_array)
                    idx = np.argmax(preds[0])
                    conf = np.max(preds[0]) * 100
                    result = class_names[str(idx)]
                    
                    # Sonuçları Göster
                    st.success(f"🎯 **Teşhis:** {result}")
                    st.info(f"📊 **Doğruluk Oranı:** %{conf:.2f}")
                    st.warning(TEDAVI_REHBERI.get(result, TEDAVI_REHBERI["default"]))
                except Exception as e:
                    st.error(f"Analiz hatası: {e}")
