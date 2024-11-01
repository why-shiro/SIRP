import numpy as np
from sklearn.model_selection import train_test_split
from model import create_road_detection_ann

# .npz dosyasını yükleyin
data = np.load('road_data.npz')
features = data['features']
labels = data['labels']

# Veriyi eğitim ve test setine ayırın
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Modeli oluşturun
model = create_road_detection_ann(input_dim=features.shape[1])

# Modeli eğitin
history = model.fit(X_train, y_train, epochs=12, batch_size=32, validation_split=0.1)

# Modeli test edin
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Eğitilmiş modeli kaydedin
model.save('road_detection_model.h5')
