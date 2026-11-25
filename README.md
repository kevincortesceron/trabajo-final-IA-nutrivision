# NutriVision — Clasificación de alimentos con asistente nutricional

Proyecto Final de Inteligencia Artificial · **Grupo 8** · Tema 8: *Clasificación de Alimentos para Asistentes de Nutrición*
Fundación Universitaria de Popayán · Ingeniería de Sistemas · 2025-2

NutriVision reconoce un plato en una foto o en video en vivo, estima sus calorías y macronutrientes, lleva el control
calórico del día y responde preguntas de nutrición con un asistente generativo.

```
Cámara/foto ──► OpenCV (CLAHE + detección del plato) ──► CNN EfficientNetB0 / ViT-B/16 ──► clase (101)
                                                                                      │
                     Base nutricional (101 platos) ◄─────────────────────────────────┘
                              │                       Diario calórico (meta Mifflin-St Jeor)
                              ▼                              │
                     NutriBot: RAG + LLM local (Qwen2.5-1.5B-Instruct) ◄──────┘
```

## Estructura

| Ruta | Contenido |
|---|---|
| `notebooks/01_entrenamiento_cnn_efficientnet.ipynb` | Entrenamiento de la CNN en Food-101 (Colab GPU): 2 fases, métricas, matriz de confusión, exportación |
| `notebooks/02_transformer_vit.ipynb` | Evaluación del Vision Transformer, mapas de atención, afinado opcional y comparación con la CNN |
| `notebooks/03_asistente_nutricional.ipynb` | Demostración del asistente generativo (RAG + LLM) |
| `src/vision.py` | OpenCV: mejora de iluminación, detección del plato (Hough + contornos), recorte, dibujo |
| `src/clasificador.py` | Interfaz común para la CNN (Keras) y el ViT (Hugging Face) |
| `src/nutricion.py` | Base nutricional, diario calórico y meta calórica |
| `src/asistente.py` | Asistente NutriBot: recuperación de contexto + generación (Transformers, Ollama o reglas) |
| `src/tiempo_real.py` | Detección en tiempo real con la cámara web (ventana OpenCV) |
| `app_streamlit.py` | App web: detección, diario y chat |
| `data/nutricion_food101.csv` | Calorías y macronutrientes aproximados por porción típica de las 101 clases |
| `models/` | Aquí van `food101_efficientnetb0.keras` y `clases.json` generados por el notebook 01 |
| `resultados/` | Métricas (`metricas_cnn.json`, `metricas_vit.json`) y figuras de los notebooks |
| `tests/test_pipeline.py` | Pruebas del flujo completo con un modelo de la misma arquitectura |

## Cómo ejecutarlo

1. **Entrenar la CNN** — abrir `notebooks/01_...ipynb` en Google Colab con GPU T4 y ejecutar todo (≈2 h). Descargar
   `entregable_cnn.zip` y copiar `models/` y `resultados/` en el proyecto.
2. **Evaluar el Transformer** — ejecutar `notebooks/02_transformer_vit.ipynb` en Colab (≈15 min). Copiar `resultados/`.
3. **Instalar localmente**
   ```bash
   python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux/Mac: source .venv/bin/activate)
   pip install -r requirements-local.txt
   ```
4. **App web**: `streamlit run app_streamlit.py`
5. **Tiempo real con la cámara**: `python -m src.tiempo_real` (teclas: `r` registrar, `p` preguntar, `q` salir)
6. **Pruebas**: `python tests/test_pipeline.py`

Sin la CNN entrenada la app usa directamente el ViT (`nateraw/food`), que se descarga de Hugging Face la primera vez.

### Motor del asistente
- `transformers` (por defecto): Qwen2.5-1.5B-Instruct local, sin API key. ≈3 GB; en CPU tarda varios segundos por respuesta.
- `ollama`: instalar [Ollama](https://ollama.com) y ejecutar `ollama pull llama3.2`.
- `reglas`: sin LLM, respuestas por plantilla (respaldo automático si el motor elegido falla).

## Despliegue en Streamlit Community Cloud
- `requirements.txt` es la versión liviana para la nube: OpenCV *headless*, PyTorch CPU y el ViT.
- En *Advanced settings* elegir **Python 3.12**.
- El asistente arranca con el motor `reglas`: Qwen2.5-1.5B no cabe en la memoria de la nube gratuita.
- La cámara en vivo con ventana de OpenCV (`src/tiempo_real.py`) solo funciona en local; en la nube se usa la
  cámara del navegador desde la app.

## Aviso
Los valores nutricionales son aproximados para porciones típicas y sirven como referencia educativa. El sistema no
reemplaza la orientación de un profesional de la salud.
