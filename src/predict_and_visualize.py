import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage import io
from data_preparation import extract_3x3_window_features

# Eğitilmiş modeli yükleyin
model = load_model('road_detection_model.h5')

# Tahmin yapmak istediğiniz görüntüyü yükleyin
image = io.imread('result.jpg')

# Görüntüden 3x3 özellikleri çıkarın
features = extract_3x3_window_features(image)

# Modelle tahmin yapın
predictions = model.predict(features)

# Tahmin edilen değerleri ikili (0 ve 1) sınıflara dönüştürün
predicted_labels = (predictions > 0.7).astype(int)

# Sonuçları orijinal görüntü boyutlarına geri dönüştürme
height, width, _ = image.shape
predicted_map = predicted_labels.reshape(height, width)

# Görselleştirme
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Orijinal görüntüyü göster
ax[0].imshow(image)
ax[0].set_title("Orijinal Görüntü")
ax[0].axis('off')

# Tahmin edilen yol haritasını göster
ax[1].imshow(predicted_map, cmap='gray')
ax[1].set_title("Tahmin Edilen Yol Haritası")
ax[1].axis('off')

plt.show()
