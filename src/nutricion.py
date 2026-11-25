"""Base de conocimiento nutricional y control calórico diario."""
from __future__ import annotations

import csv
import difflib
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from .config import RUTA_NUTRICION


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c)).replace("_", " ")


class BaseNutricional:
    """Tabla de 101 alimentos (clases de Food-101) con calorías y macronutrientes por porción.

    Los valores son aproximados, de referencia, para porciones típicas; no reemplazan una
    medición ni la valoración de un profesional en nutrición.
    """

    def __init__(self, ruta=RUTA_NUTRICION):
        with open(ruta, encoding="utf-8") as f:
            self.filas = {r["clase"]: self._tipar(r) for r in csv.DictReader(f)}
        self._indice = {}
        for clase, r in self.filas.items():
            self._indice[_normalizar(clase)] = clase
            self._indice[_normalizar(r["nombre_es"])] = clase

    @staticmethod
    def _tipar(r: dict) -> dict:
        for k in ("porcion_g", "kcal", "proteina_g", "carbohidratos_g", "grasa_g"):
            r[k] = float(r[k])
        return r

    def __getitem__(self, clase: str) -> dict:
        return self.filas[clase]

    def nombre(self, clase: str) -> str:
        return self.filas.get(clase, {}).get("nombre_es", clase.replace("_", " "))

    def buscar_en_texto(self, texto: str, max_resultados: int = 5) -> list[str]:
        """Encuentra alimentos mencionados en una pregunta (coincidencia exacta o difusa)."""
        t = _normalizar(texto)
        encontrados = [clase for nombre, clase in self._indice.items() if nombre in t]
        if not encontrados:
            palabras = [p for p in t.split() if len(p) > 3]
            for p in palabras:
                for m in difflib.get_close_matches(p, self._indice.keys(), n=2, cutoff=0.8):
                    encontrados.append(self._indice[m])
        return list(dict.fromkeys(encontrados))[:max_resultados]

    def alternativas_ligeras(self, clase: str, n: int = 3) -> list[dict]:
        """Platos de la misma categoría con menos calorías por porción."""
        base = self.filas[clase]
        candidatos = [r for r in self.filas.values()
                      if r["categoria"] == base["categoria"] and r["kcal"] < base["kcal"]]
        return sorted(candidatos, key=lambda r: r["kcal"])[:n]

    def ficha(self, clase: str, factor: float = 1.0) -> str:
        r = self.filas[clase]
        return (f"{r['nombre_es']} ({r['porcion']}, ~{r['porcion_g'] * factor:.0f} g): "
                f"{r['kcal'] * factor:.0f} kcal, proteína {r['proteina_g'] * factor:.0f} g, "
                f"carbohidratos {r['carbohidratos_g'] * factor:.0f} g, grasa {r['grasa_g'] * factor:.0f} g. "
                f"Categoría: {r['categoria']}.")


@dataclass
class Consumo:
    clase: str
    factor: float
    hora: str
    kcal: float
    proteina_g: float
    carbohidratos_g: float
    grasa_g: float


@dataclass
class DiarioCalorico:
    """Registro de lo que el usuario come en el día, con meta calórica opcional."""
    base: BaseNutricional
    meta_kcal: float | None = None
    consumos: list[Consumo] = field(default_factory=list)

    def registrar(self, clase: str, factor: float = 1.0) -> Consumo:
        r = self.base[clase]
        c = Consumo(clase, factor, datetime.now().strftime("%H:%M"),
                    r["kcal"] * factor, r["proteina_g"] * factor,
                    r["carbohidratos_g"] * factor, r["grasa_g"] * factor)
        self.consumos.append(c)
        return c

    def totales(self) -> dict:
        t = {"kcal": 0.0, "proteina_g": 0.0, "carbohidratos_g": 0.0, "grasa_g": 0.0}
        for c in self.consumos:
            for k in t:
                t[k] += getattr(c, k)
        return t

    def resumen(self) -> str:
        if not self.consumos:
            return "El usuario no ha registrado alimentos hoy."
        t = self.totales()
        lineas = [f"- {c.hora}: {self.base.nombre(c.clase)} x{c.factor:g} = {c.kcal:.0f} kcal"
                  for c in self.consumos]
        txt = "Registro de hoy:\n" + "\n".join(lineas)
        txt += (f"\nTotal: {t['kcal']:.0f} kcal (P {t['proteina_g']:.0f} g, "
                f"C {t['carbohidratos_g']:.0f} g, G {t['grasa_g']:.0f} g).")
        if self.meta_kcal:
            txt += f" Meta diaria: {self.meta_kcal:.0f} kcal; restante: {self.meta_kcal - t['kcal']:.0f} kcal."
        return txt


def meta_calorica(sexo: str, edad: int, peso_kg: float, estatura_cm: float, actividad: str) -> float:
    """Gasto energético diario estimado con la ecuación de Mifflin-St Jeor x factor de actividad."""
    tmb = 10 * peso_kg + 6.25 * estatura_cm - 5 * edad + (5 if sexo.lower() in ("masculino", "hombre", "m") else -161)
    factores = {"sedentaria": 1.2, "ligera": 1.375, "moderada": 1.55, "alta": 1.725}
    return tmb * factores.get(actividad, 1.2)
