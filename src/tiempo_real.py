"""Detección y clasificación de alimentos en tiempo real con la cámara web (OpenCV).

Uso:
    python -m src.tiempo_real               # modelo automático (CNN si existe, si no ViT)
    python -m src.tiempo_real --modelo vit  # forzar el Transformer
Teclas: [r] registrar el plato en el diario  [p] preguntar al asistente  [q] salir
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, deque

import cv2

from .asistente import AsistenteNutricional
from .clasificador import cargar_clasificador
from .config import UMBRAL_CONFIANZA
from .nutricion import BaseNutricional, DiarioCalorico
from .vision import detectar_plato, dibujar_resultado, mejorar_iluminacion, preparar_para_modelo, recortar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="auto", choices=["auto", "cnn", "vit"])
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--cada", type=int, default=5, help="clasificar 1 de cada N cuadros")
    ap.add_argument("--motor", default="reglas", choices=["transformers", "ollama", "reglas"])
    args = ap.parse_args()

    clf = cargar_clasificador(args.modelo)
    base = BaseNutricional()
    diario = DiarioCalorico(base)
    asistente = AsistenteNutricional(base, diario, motor=args.motor)
    print(f"Modelo de visión: {clf.nombre}")

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    votos = deque(maxlen=7)  # suavizado temporal: voto mayoritario de las últimas predicciones
    caja, top = None, [("...", 0.0)]
    n, t0, fps = 0, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % args.cada == 0:
            mejorado = mejorar_iluminacion(frame)
            caja = detectar_plato(mejorado)
            top = clf.predecir(preparar_para_modelo(recortar(mejorado, caja)))
            votos.append(top[0][0])
            fps = args.cada / (time.time() - t0)
            t0 = time.time()

        if caja is not None:
            clase = Counter(votos).most_common(1)[0][0]
            conf = max((p for c, p in top if c == clase), default=top[0][1])
            frame = dibujar_resultado(frame, caja, base.nombre(clase), conf, base[clase]["kcal"])
            for i, (c, p) in enumerate(top):
                cv2.putText(frame, f"{i + 1}. {base.nombre(c)} {p:.0%}", (10, 25 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if conf < UMBRAL_CONFIANZA:
                cv2.putText(frame, "Prediccion dudosa: acerque el plato", (10, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        total = diario.totales()["kcal"]
        cv2.putText(frame, f"Hoy: {total:.0f} kcal | {fps:.1f} pred/s", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("NutriVision - Grupo 8 (q: salir, r: registrar, p: preguntar)", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("r") and votos:
            c = diario.registrar(Counter(votos).most_common(1)[0][0])
            print(f"Registrado: {base.nombre(c.clase)} ({c.kcal:.0f} kcal)")
        if tecla == ord("p") and votos:
            asistente.ultimo_detectado = Counter(votos).most_common(1)[0][0]
            pregunta = input("Pregunta para NutriBot: ")
            print("NutriBot:", asistente.responder(pregunta))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
