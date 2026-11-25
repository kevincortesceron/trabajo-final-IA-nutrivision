"""Procesamiento de imagen con OpenCV: detección de la región del plato y preprocesamiento.

La clasificación la hacen las redes neuronales (clasificador.py); este módulo se encarga de
la etapa de *detección*: localizar el plato/alimento dentro de la escena para recortarlo antes
de clasificar, y de mejorar la imagen (iluminación) para que el modelo reciba entradas limpias.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import TAM_IMAGEN


def mejorar_iluminacion(img_bgr: np.ndarray) -> np.ndarray:
    """Ecualiza el canal de luminancia con CLAHE (útil en fotos de restaurante o cocina oscura)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def detectar_plato(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Devuelve la caja (x, y, w, h) del plato/alimento principal.

    Estrategia en dos pasos:
      1. Transformada de Hough para círculos: la mayoría de platos son circulares vistos desde arriba.
      2. Si no hay círculo, se toma el contorno más grande del mapa de bordes (Canny + dilatación).
    Si ninguna funciona, se usa un recorte central cuadrado (el alimento suele estar centrado).
    """
    h, w = img_bgr.shape[:2]
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.medianBlur(gris, 5)
    lado_min = min(h, w)

    circulos = cv2.HoughCircles(
        gris, cv2.HOUGH_GRADIENT, dp=1.2, minDist=lado_min // 2,
        param1=100, param2=60, minRadius=int(lado_min * 0.25), maxRadius=int(lado_min * 0.55),
    )
    if circulos is not None:
        cx, cy, r = np.round(circulos[0, 0]).astype(int)
        x0, y0 = max(cx - r, 0), max(cy - r, 0)
        x1, y1 = min(cx + r, w), min(cy + r, h)
        if (x1 - x0) * (y1 - y0) > 0.15 * h * w:
            return x0, y0, x1 - x0, y1 - y0

    bordes = cv2.Canny(gris, 50, 150)
    bordes = cv2.dilate(bordes, np.ones((7, 7), np.uint8), iterations=2)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        c = max(contornos, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        if 0.15 * h * w < cw * ch < 0.95 * h * w:
            return x, y, cw, ch

    lado = int(lado_min * 0.8)
    return (w - lado) // 2, (h - lado) // 2, lado, lado


def recortar(img_bgr: np.ndarray, caja: tuple[int, int, int, int], margen: float = 0.05) -> np.ndarray:
    x, y, w, h = caja
    H, W = img_bgr.shape[:2]
    dx, dy = int(w * margen), int(h * margen)
    return img_bgr[max(y - dy, 0):min(y + h + dy, H), max(x - dx, 0):min(x + w + dx, W)]


def preparar_para_modelo(img_bgr: np.ndarray, tam: int = TAM_IMAGEN) -> np.ndarray:
    """BGR (OpenCV) -> RGB redimensionado a tam x tam, uint8. La normalización la hace cada modelo."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (tam, tam), interpolation=cv2.INTER_AREA)


def dibujar_resultado(img_bgr: np.ndarray, caja, etiqueta: str, confianza: float,
                      kcal: float | None = None) -> np.ndarray:
    """Dibuja la caja del plato y la etiqueta con la confianza y las calorías estimadas."""
    out = img_bgr.copy()
    x, y, w, h = caja
    color = (60, 180, 75) if confianza >= 0.4 else (0, 165, 255)
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
    texto = f"{etiqueta} {confianza:.0%}" + (f" | ~{kcal:.0f} kcal" if kcal else "")
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y_txt = max(y, th + 12)
    cv2.rectangle(out, (x, y_txt - th - 12), (x + tw + 10, y_txt), color, -1)
    cv2.putText(out, texto, (x + 5, y_txt - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
