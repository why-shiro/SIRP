import numpy as np
import os
from skimage import io

def normalize_rgb(image):
    """
    Normalizes RGB values of the image to range [0, 1].
    """
    return image / 255.0

def extract_3x3_window_features(image):
    """
    Extracts 3x3 window RGB features for each pixel in the image.
    Pads the image with edge values to handle border pixels.
    """
    image = normalize_rgb(image)
    height, width, _ = image.shape
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    features = []

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            window = padded_image[i-1:i+2, j-1:j+2, :]
            flattened_window = window.flatten()
            features.append(flattened_window)
    
    return np.array(features)

def prepare_labels(reference_map):
    """
    Extracts labels from the reference map where roads are labeled as 1 and non-roads as 0.
    """
    return reference_map.flatten()  # Flatten the 2D reference map to a 1D array

def load_images_from_directory(directory, file_extension='.png'):
    """
    Loads all images with the given extension in the specified directory.
    """
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(file_extension):
            image = io.imread(os.path.join(directory, filename))
            images.append(image)
            filenames.append(filename)
    return images, filenames

def save_data(features, labels, filename='road_data.npz'):
    """
    Saves features and labels to a .npz file.
    """
    np.savez(filename, features=features, labels=labels)
    print(f"Data saved to {filename}")

# Veriyi hazırlama ve kaydetme işlemi
def prepare_and_save_data(image_dir, reference_dir, output_filename='road_data.npz'):
    # Görüntüleri ve dosya isimlerini yükleyin
    images, filenames = load_images_from_directory(image_dir)
    
    # Özellikleri ve etiketleri çıkarın
    all_features = []
    all_labels = []
    
    for image, filename in zip(images, filenames):
        # Görüntü özelliklerini çıkar
        features = extract_3x3_window_features(image)
        all_features.append(features)
        
        # Referans haritayı yükleyin ve etiketleri çıkarın
        reference_map_path = os.path.join(reference_dir, filename)
        if os.path.exists(reference_map_path):
            reference_map = io.imread(reference_map_path, as_gray=True)
            labels = prepare_labels((reference_map > 0.5).astype(int))
            all_labels.append(labels)
        else:
            print(f"Warning: Reference map not found for {filename}. Skipping this image.")
    
    # Tüm özellikleri ve etiketleri birleştirin
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Veriyi kaydedin
    save_data(all_features, all_labels, output_filename)

# Örnek kullanım
prepare_and_save_data('output/converter/', 'referance_map/processed_road_map/', 'road_data.npz')

# .npz dosyasını yükleyin
data = np.load('road_data.npz')

# Özellikleri ve etiketleri alın
features = data['features']
labels = data['labels']

# Veri hakkında bilgi edinmek için
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# İlk birkaç örneğe göz atmak için
print("First few features:", features[:5])
print("First few labels:", labels[:5])

