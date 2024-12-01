import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuración de parámetros
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Dimensiones de las imágenes
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = os.path.join("models", "detector_model.h5")

# Configuración de Data Generators
def create_data_generators(data_dir):
    datagen = ImageDataGenerator(
        rescale=1.0/255,                # Normalización de valores de píxeles
        validation_split=0.2,           # División en entrenamiento/validación
        rotation_range=10,              # Rotación aleatoria
        width_shift_range=0.1,          # Desplazamiento horizontal
        height_shift_range=0.1,         # Desplazamiento vertical
        zoom_range=0.1                  # Zoom aleatorio
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, val_generator

# Función principal de entrenamiento
def train_detector():
    data_dir = "data"  # Ruta a los datos de imágenes organizados en subcarpetas
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directorio de datos no encontrado: {data_dir}")

    print("Cargando datos de entrenamiento y validación...")
    train_generator, val_generator = create_data_generators(data_dir)

    print("Construyendo el modelo CNN...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Clasificación binaria
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Entrenando el modelo CNN...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    print("Guardando el modelo entrenado...")
    model.save(MODEL_PATH)
    print(f"Modelo CNN guardado en {MODEL_PATH}")

if __name__ == "__main__":
    train_detector()
