import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# Configuración de parámetros
VOCAB_SIZE = 100  # Tamaño del vocabulario
EMBEDDING_DIM = 64  # Dimensión de los embeddings
LSTM_UNITS = 128  # Unidades de LSTM
MAX_SEQ_LENGTH = 20  # Longitud máxima de las secuencias
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = os.path.join("models", "ocr_model.h5")

# Simulación de datos (reemplazar con datos reales de etiquetas)
def generate_dummy_data(num_samples=1000):
    np.random.seed(42)
    sequences = [np.random.randint(1, VOCAB_SIZE, MAX_SEQ_LENGTH) for _ in range(num_samples)]
    labels = [np.random.randint(0, VOCAB_SIZE) for _ in range(num_samples)]
    return np.array(sequences), np.array(labels)

# Función principal de entrenamiento
def train_ocr():
    print("Generando datos de entrenamiento...")
    x_train, y_train = generate_dummy_data(num_samples=800)
    x_val, y_val = generate_dummy_data(num_samples=200)

    print("Construyendo el modelo OCR...")
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(0.2),
        Dense(VOCAB_SIZE, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Entrenando el modelo OCR...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    print("Guardando el modelo entrenado...")
    model.save(MODEL_PATH)
    print(f"Modelo OCR guardado en {MODEL_PATH}")

if __name__ == "__main__":
    train_ocr()
