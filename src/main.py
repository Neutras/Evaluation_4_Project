import tensorflow as tf
import cv2
import pytesseract
import numpy as np
import os

# Configurar la ruta de Tesseract si estás en Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ruta a los modelos entrenados
DETECTOR_MODEL_PATH = os.path.join("models", "detector_model.h5")

# Cargar modelos
def load_detector_model():
    print("Cargando modelo detector...")
    if not os.path.exists(DETECTOR_MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {DETECTOR_MODEL_PATH}")
    return tf.keras.models.load_model(DETECTOR_MODEL_PATH)

# Preprocesar imagen para el detector
def preprocess_for_detection(image_path):
    print(f"Preprocesando la imagen {image_path} para detección...")
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0  # Normalizar
    return np.expand_dims(img_normalized, axis=0)

# Realizar detección de errores en etiquetas
def detect_label_error(detector_model, image_path):
    processed_image = preprocess_for_detection(image_path)
    prediction = detector_model.predict(processed_image)
    return prediction[0][0] > 0.5  # Retorna True si se detecta un error

# Extraer texto de la etiqueta usando OCR
def extract_text_with_ocr(image_path):
    print("Extrayendo texto con OCR...")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='eng')
    return text

# Flujo principal
def main(image_path):
    # Cargar modelo detector
    detector_model = load_detector_model()

    # Detectar errores en la etiqueta
    print("Detectando errores en la etiqueta...")
    is_error = detect_label_error(detector_model, image_path)
    if is_error:
        print("Error detectado en la etiqueta.")
    else:
        print("No se detectaron errores en la etiqueta.")

    # Extraer texto de la etiqueta
    text = extract_text_with_ocr(image_path)
    print("Texto extraído de la etiqueta:")
    print(text)

if __name__ == "__main__":
    # Ruta a la imagen a analizar
    IMAGE_PATH = "data/test_image.jpg"  # Cambia esto por tu imagen de prueba
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Imagen no encontrada en {IMAGE_PATH}")

    main(IMAGE_PATH)
