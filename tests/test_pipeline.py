"""Pruebas de humo del flujo completo sin GPU ni dataset.

Construye la misma arquitectura del notebook 01 con pesos aleatorios para verificar que
visión (OpenCV) -> clasificador -> base nutricional -> diario -> asistente funcionan juntos.
Ejecutar desde la raíz del proyecto:  python -m pytest -q tests  (o python tests/test_pipeline.py)
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asistente import AsistenteNutricional  # noqa: E402
from src.nutricion import BaseNutricional, DiarioCalorico, meta_calorica  # noqa: E402
from src.vision import detectar_plato, dibujar_resultado, mejorar_iluminacion, preparar_para_modelo, recortar  # noqa: E402


def imagen_sintetica():
    img = np.full((480, 640, 3), (40, 90, 140), np.uint8)          # mesa
    cv2.circle(img, (330, 250), 170, (235, 235, 235), -1)           # plato
    cv2.circle(img, (330, 250), 90, (30, 120, 200), -1)             # comida
    return img


def test_base_nutricional():
    base = BaseNutricional()
    assert len(base.filas) == 101
    assert base.buscar_en_texto("¿cuántas calorías tiene la lasaña?") == ["lasagna"]
    assert "ramen" in base.buscar_en_texto("me comí un ramen")
    assert all(a["kcal"] < base["pizza"]["kcal"] for a in base.alternativas_ligeras("pizza"))


def test_vision():
    img = imagen_sintetica()
    caja = detectar_plato(mejorar_iluminacion(img))
    x, y, w, h = caja
    assert abs((x + w / 2) - 330) < 40 and abs((y + h / 2) - 250) < 40, caja
    x_in = preparar_para_modelo(recortar(img, caja))
    assert x_in.shape == (224, 224, 3) and x_in.dtype == np.uint8
    assert dibujar_resultado(img, caja, "Pizza", 0.9, 565).shape == img.shape


def test_diario_y_asistente():
    base = BaseNutricional()
    diario = DiarioCalorico(base, meta_kcal=meta_calorica("femenino", 21, 60, 165, "ligera"))
    diario.registrar("pizza", 1.5)
    diario.registrar("caesar_salad")
    assert round(diario.totales()["kcal"]) == round(565 * 1.5 + 360)
    bot = AsistenteNutricional(base, diario, motor="reglas")
    bot.ultimo_detectado = "pizza"
    assert "Total" in bot.responder("¿cuántas calorías llevo hoy?")
    assert "Alternativas" in bot.responder("¿algo más ligero que la pizza?")
    ctx, clases = bot.construir_contexto("y el ceviche?")
    assert clases[:2] == ["pizza", "ceviche"] and "Registro de hoy" in ctx


def test_clasificador_cnn_misma_arquitectura():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    from src.clasificador import ClasificadorCNN

    base_net = keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3))
    ent = keras.Input((224, 224, 3))
    x = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),
                          layers.RandomZoom(0.15), layers.RandomContrast(0.15)])(ent)
    x = base_net(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    sal = layers.Dense(101, activation="softmax", dtype="float32")(x)
    modelo = keras.Model(ent, sal)
    clases = list(BaseNutricional().filas)
    with tempfile.TemporaryDirectory() as d:
        modelo.save(f"{d}/m.keras")
        json.dump(clases, open(f"{d}/c.json", "w"))
        clf = ClasificadorCNN(f"{d}/m.keras", f"{d}/c.json")
        top = clf.predecir(preparar_para_modelo(imagen_sintetica()))
    assert len(top) == 3 and all(c in clases for c, _ in top)
    assert abs(sum(p for _, p in top)) <= 1.0 + 1e-5
    print("parámetros:", modelo.count_params(), "| tf", tf.__version__)


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            print("OK", nombre)
