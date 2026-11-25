"""Asistente nutricional generativo con recuperación de contexto (RAG ligero).

Flujo de cada pregunta:
  1. Recuperación: se buscan en la base nutricional los alimentos mencionados en la pregunta,
     el último alimento detectado por la cámara y el registro calórico del día.
  2. Aumento: ese contexto se inserta en el prompt junto con reglas de seguridad.
  3. Generación: un LLM local responde en español apoyándose SOLO en los datos recuperados.

Motores disponibles (todos en Python, sin API keys):
  - "transformers": Qwen2.5-1.5B-Instruct vía Hugging Face Transformers (por defecto).
  - "ollama": cualquier modelo servido por Ollama en localhost.
  - "reglas": respuestas por plantillas, sin LLM (respaldo si no hay recursos para el modelo).
"""
from __future__ import annotations

import json
import urllib.request

from .config import LLM_HF, LLM_OLLAMA, URL_OLLAMA
from .nutricion import BaseNutricional, DiarioCalorico

PROMPT_SISTEMA = """Eres NutriBot, un asistente de nutrición que acompaña a una app de visión por computador \
que reconoce platos de comida. Responde en español, de forma breve, clara y amable.
Reglas:
- Usa los datos nutricionales del CONTEXTO para cifras de calorías y macronutrientes; no inventes cifras.
- Aclara que los valores son aproximados y dependen de la porción y la preparación.
- Puedes dar recomendaciones generales de alimentación saludable (equilibrio, porciones, alternativas).
- No diagnostiques enfermedades ni prescribas dietas clínicas; ante condiciones médicas \
(diabetes, embarazo, alergias, trastornos alimentarios) recomienda consultar a un profesional de la salud."""


class AsistenteNutricional:
    def __init__(self, base: BaseNutricional, diario: DiarioCalorico, motor: str = "transformers"):
        self.base = base
        self.diario = diario
        self.motor = motor
        self.historial: list[dict] = []
        self.ultimo_detectado: str | None = None
        self._pipe = None

    # ---------- 1. Recuperación ----------
    def construir_contexto(self, pregunta: str) -> tuple[str, list[str]]:
        clases = self.base.buscar_en_texto(pregunta)
        if self.ultimo_detectado and self.ultimo_detectado not in clases:
            clases.insert(0, self.ultimo_detectado)
        partes = []
        if self.ultimo_detectado:
            partes.append(f"Último plato detectado por la cámara: {self.base.nombre(self.ultimo_detectado)}.")
        for c in clases:
            partes.append("Dato: " + self.base.ficha(c))
            alt = self.base.alternativas_ligeras(c)
            if alt:
                partes.append("  Alternativas más ligeras de la misma categoría: " +
                              ", ".join(f"{a['nombre_es']} ({a['kcal']:.0f} kcal)" for a in alt))
        partes.append(self.diario.resumen())
        return "\n".join(partes), clases

    # ---------- 2 y 3. Aumento + generación ----------
    def responder(self, pregunta: str) -> str:
        contexto, clases = self.construir_contexto(pregunta)
        if self.motor == "reglas":
            respuesta = self._responder_reglas(pregunta, clases)
        else:
            mensajes = [{"role": "system", "content": PROMPT_SISTEMA + "\n\nCONTEXTO:\n" + contexto}]
            mensajes += self.historial[-6:]
            mensajes.append({"role": "user", "content": pregunta})
            try:
                respuesta = self._generar_ollama(mensajes) if self.motor == "ollama" else self._generar_hf(mensajes)
            except Exception as e:  # sin modelo o sin servidor: se degrada a reglas
                respuesta = (self._responder_reglas(pregunta, clases) +
                             f"\n\n_(Motor '{self.motor}' no disponible: {type(e).__name__}. Respuesta por reglas.)_")
        self.historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": respuesta}]
        return respuesta

    def _generar_hf(self, mensajes: list[dict]) -> str:
        if self._pipe is None:
            import torch
            from transformers import pipeline

            self._pipe = pipeline("text-generation", model=LLM_HF,
                                  torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                  device_map="auto")
        salida = self._pipe(mensajes, max_new_tokens=300, do_sample=True, temperature=0.4, top_p=0.9)
        return salida[0]["generated_text"][-1]["content"].strip()

    def _generar_ollama(self, mensajes: list[dict]) -> str:
        cuerpo = json.dumps({"model": LLM_OLLAMA, "messages": mensajes, "stream": False,
                             "options": {"temperature": 0.4}}).encode()
        req = urllib.request.Request(URL_OLLAMA, data=cuerpo, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"].strip()

    def _responder_reglas(self, pregunta: str, clases: list[str]) -> str:
        p = pregunta.lower()
        if any(w in p for w in ("hoy", "llevo", "total", "día", "dia", "restante", "meta")):
            return self.diario.resumen()
        if not clases:
            return ("No identifiqué un alimento en tu pregunta. Toma una foto del plato o menciona su nombre "
                    "(por ejemplo: '¿cuántas calorías tiene la lasaña?').")
        c = clases[0]
        r = self.base[c]
        txt = self.base.ficha(c)
        if any(w in p for w in ("ligero", "saludable", "alternativa", "cambiar", "reemplaz", "menos")):
            alt = self.base.alternativas_ligeras(c)
            txt += ("\nAlternativas más ligeras: " + ", ".join(f"{a['nombre_es']} ({a['kcal']:.0f} kcal)" for a in alt)
                    if alt else "\nYa es de las opciones más ligeras de su categoría.")
        elif r["kcal"] >= 600:
            txt += "\nEs un plato denso en energía: considera media porción o acompañarlo con verduras."
        return txt + "\n(Valores aproximados; varían según porción y preparación.)"
