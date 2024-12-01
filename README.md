
# Evaluación 4: Sistema de Verificación de Etiquetas con IA

Este proyecto implementa un sistema basado en Deep Learning para verificar la legibilidad y precisión de etiquetas de productos. Se enfoca en detectar errores comunes en fechas de envasado mediante una combinación de detección de características visuales y reconocimiento de texto (OCR).

## Características Principales
- **Preprocesamiento de Imágenes**: Ajustes de contraste, brillo, y binarización para mejorar la legibilidad.
- **Modelo CNN**: Para detectar errores visuales en las etiquetas.
- **Modelo OCR**: Para extraer y validar texto en las etiquetas.
- **Integración Completa**: Pipeline que une detección y reconocimiento.

## Requisitos
- Python 3.9+
- Librerías listadas en `requirements.txt`.

## Estructura del Proyecto
```
Evaluation_4_Project/
│
├── src/                  # Código fuente
├── data/                 # Datos de entrenamiento y pruebas
├── models/               # Modelos entrenados
├── docs/                 # Documentación adicional
├── config/               # Configuraciones del proyecto
└── README.md             # Información del proyecto
```

## Instrucciones de Uso
1. Clonar el repositorio:
   ```
   git clone <url_del_repositorio>
   cd Evaluation_4_Project
   ```
2. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Entrenar los modelos:
   ```
   python src/train_detector.py
   python src/train_ocr.py
   ```
4. Ejecutar el sistema completo:
   ```
   python src/main.py
   ```
