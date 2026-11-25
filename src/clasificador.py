"""Clasificadores de alimentos: CNN (EfficientNetB0 propio) y Transformer (ViT-B/16 afinado en Food-101).

Ambos exponen la misma interfaz `predecir(img_rgb_224) -> [(clase, probabilidad), ...]`,
por lo que la app y el script de tiempo real pueden cambiar de modelo sin tocar el resto del código.
"""
from __future__ import annotations

import json

import numpy as np

from .config import MODELO_VIT_HF, RUTA_CLASES, RUTA_MODELO_CNN, TOP_K


class ClasificadorCNN:
    """Carga el modelo Keras entrenado en notebooks/01_entrenamiento_cnn_efficientnet.ipynb."""

    nombre = "CNN EfficientNetB0 (transfer learning, propio)"

    def __init__(self, ruta_modelo=RUTA_MODELO_CNN, ruta_clases=RUTA_CLASES):
        import tensorflow as tf  # import diferido: la app funciona aunque solo se use ViT

        self.modelo = tf.keras.models.load_model(ruta_modelo)
        with open(ruta_clases, encoding="utf-8") as f:
            self.clases = json.load(f)

    def predecir(self, img_rgb: np.ndarray, k: int = TOP_K) -> list[tuple[str, float]]:
        # El preprocesamiento de EfficientNet (escala 0-255) está incluido dentro del modelo.
        x = np.expand_dims(img_rgb.astype("float32"), 0)
        prob = self.modelo.predict(x, verbose=0)[0]
        idx = np.argsort(prob)[::-1][:k]
        return [(self.clases[i], float(prob[i])) for i in idx]


class ClasificadorViT:
    """Vision Transformer (google/vit-base-patch16-224-in21k) afinado en Food-101 (nateraw/food)."""

    nombre = "Transformer ViT-B/16 (nateraw/food)"

    def __init__(self, modelo=MODELO_VIT_HF):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.torch = torch
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.procesador = AutoImageProcessor.from_pretrained(modelo)
        self.modelo = AutoModelForImageClassification.from_pretrained(modelo).to(self.dispositivo).eval()
        self.id2label = self.modelo.config.id2label

    def predecir(self, img_rgb: np.ndarray, k: int = TOP_K) -> list[tuple[str, float]]:
        entradas = self.procesador(images=img_rgb, return_tensors="pt").to(self.dispositivo)
        with self.torch.no_grad():
            logits = self.modelo(**entradas).logits[0]
        prob = self.torch.softmax(logits, -1).cpu().numpy()
        idx = np.argsort(prob)[::-1][:k]
        return [(self.id2label[int(i)], float(prob[i])) for i in idx]


def cargar_clasificador(preferencia: str = "auto"):
    """'cnn', 'vit' o 'auto' (usa la CNN propia si ya está entrenada; si no, el ViT)."""
    if preferencia == "cnn" or (preferencia == "auto" and RUTA_MODELO_CNN.exists()):
        return ClasificadorCNN()
    return ClasificadorViT()
