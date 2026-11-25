"""Configuración central del proyecto NutriVision (Grupo 8 - Proyecto Final IA)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ / "models"
DIR_DATOS = RAIZ / "data"
DIR_RESULTADOS = RAIZ / "resultados"

# Modelo CNN entrenado en el notebook 01 (EfficientNetB0 + transfer learning)
RUTA_MODELO_CNN = DIR_MODELOS / "food101_efficientnetb0.keras"
RUTA_CLASES = DIR_MODELOS / "clases.json"

# Modelo Transformer (ViT) afinado sobre Food-101, publicado en Hugging Face
MODELO_VIT_HF = "nateraw/food"

# Tabla nutricional de referencia (valores aproximados por porción típica)
RUTA_NUTRICION = DIR_DATOS / "nutricion_food101.csv"

TAM_IMAGEN = 224          # resolución de entrada de EfficientNetB0 y ViT-B/16
UMBRAL_CONFIANZA = 0.40   # por debajo de este valor se avisa que la predicción es dudosa
TOP_K = 3

# LLM local para el asistente generativo (Hugging Face Transformers, sin API key)
LLM_HF = "Qwen/Qwen2.5-1.5B-Instruct"
# Alternativa local con Ollama (https://ollama.com): `ollama pull llama3.2`
LLM_OLLAMA = "llama3.2"
URL_OLLAMA = "http://localhost:11434/api/chat"
