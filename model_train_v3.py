import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from numba import jit, cuda 



#--------------------------------------------------------------------------------------------------------------------------------------------------
# Chargement des données 
#--------------------------------------------------------------------------------------------------------------------------------------------------

# Paramètres
train_dir = './train'
test_dir = './test'
val_dir = './valid'

# Fonction pour charger les annotations CSV
def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    image_paths = df['filename'].values
    labels = df['class'].values
    xmin = df['xmin'].values
    ymin = df['ymin'].values
    xmax = df['xmax'].values
    ymax = df['ymax'].values
    return image_paths, labels, xmin, ymin, xmax, ymax

# Convertir les labels de texte en indices numériques
def label_to_index(labels, class_dict):
    return np.array([class_dict[label] for label in labels])

# Charger les données d'entraînement, de test et de validation
train_csv = os.path.join(train_dir, '_annotations.csv')
test_csv = os.path.join(test_dir, '_annotations.csv')
val_csv = os.path.join(val_dir, '_annotations.csv')

train_paths, train_labels, train_xmin, train_ymin, train_xmax, train_ymax = load_annotations(train_csv)
test_paths, test_labels, test_xmin, test_ymin, test_xmax, test_ymax = load_annotations(test_csv)
val_paths, val_labels, val_xmin, val_ymin, val_xmax, val_ymax = load_annotations(val_csv)

# Dictionnaire des classes (vous devez définir les classes selon votre dataset)
class_dict = {'bottle': 0, 'can': 1, 'bottle_cap': 2}  # Adapter selon vos classes
num_classes = len(class_dict)

# Convertir les labels en indices numériques
train_labels = label_to_index(train_labels, class_dict)
test_labels = label_to_index(test_labels, class_dict)
val_labels = label_to_index(val_labels, class_dict)

# Convertir les labels en format one-hot
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)

# Charger les images
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=(256, 256))  # Redimensionner si nécessaire
        img = img_to_array(img) / 255.0  # Normalisation des images
        images.append(img)
    return np.array(images)

train_images = load_images([os.path.join(train_dir, path) for path in train_paths])
test_images = load_images([os.path.join(test_dir, path) for path in test_paths])
val_images = load_images([os.path.join(val_dir, path) for path in val_paths])

'''# Normalisation des coordonnées des bounding boxes (entre 0 et 1)
def normalize_bbox(xmin, ymin, xmax, ymax, img_width, img_height):
    xmin_norm = xmin / img_width
    ymin_norm = ymin / img_height
    xmax_norm = xmax / img_width
    ymax_norm = ymax / img_height
    return xmin_norm, ymin_norm, xmax_norm, ymax_norm

# Appliquer la normalisation des bounding boxes pour chaque ensemble de données
train_bboxes = np.array([normalize_bbox(x, y, w, h, 256, 256) for x, y, w, h in zip(train_xmin, train_ymin, train_xmax, train_ymax)])
test_bboxes = np.array([normalize_bbox(x, y, w, h, 256, 256) for x, y, w, h in zip(test_xmin, test_ymin, test_xmax, test_ymax)])
val_bboxes = np.array([normalize_bbox(x, y, w, h, 256, 256) for x, y, w, h in zip(val_xmin, val_ymin, val_xmax, val_ymax)])'''

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Entrainement du modèle
#--------------------------------------------------------------------------------------------------------------------------------------------------


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

# Construire le modèle CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # Classification multi-classes
])

# Compiler le modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraîner le modèle
history = model.fit(
    train_images,
    train_labels,
    epochs=30,
    batch_size=32,
    validation_data=(val_images, val_labels)
)

# Sauvegarder le modèle
model.save('C:/Users/Thomas/OneDrive/Documents/IPSA/PMI/cnn_dechet_model_with_bbox.keras')

# Sauvegarder les poids
model.save_weights('C:/Users/Thomas/OneDrive/Documents/IPSA/PMI/cnn_dechet_weights.weights.h5')

# Sauvegarder la configuration
with open('C:/Users/Thomas/OneDrive/Documents/IPSA/PMI/cnn_dechet_config.json', 'w') as f:
    f.write(model.to_json())

# Tracer la courbe de précision
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Tracer la courbe de perte
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()