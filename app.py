import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Sayfa ayarları ve başlık
st.set_page_config(page_title="Bitki Doktoru AI", page_icon="🌿", layout="centered")
st.title("🌿 Akıllı Bitki Hastalık Teşhis Sistemi")
st.markdown("---")

# Model ve Etiketleri Yükleme
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model('en_iyi_yaprak_modeli.h5')
    with open('sinif_isimleri.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)
    return model, labels

model, class_names = load_model_and_labels()

# Tedavi Bilgi Bankası (Buraya her hastalık için öneri ekleyebilirsin)
TEDAVI_REHBERI = {
    "Tomato___Tomato_YellowLeaf__Curl_Virus": "🛑 **Öneri:** Hastalıklı bitkileri derhal sökün. Beyaz sineklerle mücadele edin ve dayanıklı tohumlar tercih edin.",
    "Apple___Apple_scab": "🛑 **Öneri:** Dökülen yaprakları temizleyin. Erken ilkbaharda uygun fungisitlerle ilaçlama yapın.",
    "Corn_(maize)___Common_rust": "🛑 **Öneri:** Dayanıklı mısır çeşitleri ekin. Aşırı nemden kaçınmak için sulama zamanlamasını ayarlayın.",
    "default": "⚠️ **Öneri:** Uzman bir ziraat mühendisine danışın ve hastalıklı bölgeyi diğer bitkilerden izole edin."
}

# GİRİŞ SEÇENEKLERİ (Kamera Desteği Eklendi!)
tab1, tab2 = st.tabs(["📸 Fotoğraf Çek", "📁 Dosya Yükle"])

with tab1:
    cam_file = st.camera_input("Yaprağın canlı görüntüsünü al")

with tab2:
    uploaded_file = st.file_uploader("Veya bir dosya seçin...", type=["jpg", "jpeg", "png"])

# Hangi kaynaktan gelirse gelsin resmi işle
source_file = cam_file if cam_file is not None else uploaded_file

if source_file is not None:
    image = Image.open(source_file)
    st.image(image, caption="Analiz Edilecek Yaprak", use_container_width=True)

    if st.button("Hemen Teşhis Et"):
        with st.spinner("Yapay Zeka Analiz Ediyor..."):
            try:
                # Görüntü Ön İşleme
                img = image.convert('RGB').resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Tahmin
                predictions = model.predict(img_array)
                idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                result = class_names[str(idx)]

                # Sonuç Ekranı
                st.success(f"🎯 **Teşhis:** {result}")
                st.info(f"📊 **Güven Oranı:** %{confidence:.2f}")
                
                # Tedavi Önerisi Getir
                tavsiye = TEDAVI_REHBERI.get(result, TEDAVI_REHBERI["default"])
                st.warning(tavsiye)
                
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")