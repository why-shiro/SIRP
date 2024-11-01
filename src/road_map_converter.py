import cv2
import numpy as np
import os

# Klasör yollarını tanımlayın
input_folder = './src/referance_map/references/'  # Fotoğrafların bulunduğu klasör yolu
output_folder = './src/referance_map/processed_road_map/'  # Çıktının kaydedileceği klasör yolu

# Eğer output klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tüm resimleri işlemek için klasörü gez
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # İstediğiniz dosya formatlarını ekleyin
        # Resmi yükle
        img = cv2.imread(os.path.join(input_folder, filename))

        # Beyaz dışındaki tüm renkleri siyah yap
        lower_white = np.array([200, 200, 200])  # Beyazın alt sınırı (isteğe göre ayarlayın)
        upper_white = np.array([255, 255, 255])  # Beyazın üst sınırı

        # Maske oluştur ve beyaz olan kısımları koru, diğer yerleri siyaha çevir
        mask = cv2.inRange(img, lower_white, upper_white)
        result = cv2.bitwise_and(img, img, mask=mask)

        # Çıktıyı kaydet
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

print("İşlem tamamlandı. Tüm resimler output klasörüne kaydedildi.")
