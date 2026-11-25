"""NutriVision - Clasificación de alimentos + asistente nutricional (Grupo 8, Proyecto Final IA).

Ejecutar:  streamlit run app_streamlit.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.asistente import AsistenteNutricional
from src.clasificador import cargar_clasificador
from src.config import RUTA_MODELO_CNN, UMBRAL_CONFIANZA
from src.nutricion import BaseNutricional, DiarioCalorico, meta_calorica
from src.vision import detectar_plato, dibujar_resultado, mejorar_iluminacion, preparar_para_modelo, recortar

st.set_page_config(page_title="NutriVision", page_icon="🍽️", layout="wide")


@st.cache_resource(show_spinner="Cargando modelo de visión...")
def obtener_clasificador(preferencia: str):
    return cargar_clasificador(preferencia)


@st.cache_resource
def obtener_base():
    return BaseNutricional()


base = obtener_base()
if "diario" not in st.session_state:
    st.session_state.diario = DiarioCalorico(base)
    st.session_state.chat = []
diario: DiarioCalorico = st.session_state.diario

# ---------------- Barra lateral ----------------
with st.sidebar:
    st.title("🍽️ NutriVision")
    st.caption("Proyecto Final IA · Grupo 8")
    opciones = (["cnn"] if RUTA_MODELO_CNN.exists() else []) + ["vit"]
    modelo = st.selectbox("Modelo de visión", opciones,
                          format_func=lambda m: {"vit": "Transformer ViT-B/16",
                                                 "cnn": "CNN EfficientNetB0 (propio)"}[m])
    motor = st.selectbox("Motor del asistente", ["transformers", "ollama", "reglas"],
                         help="transformers: Qwen2.5-1.5B local · ollama: servidor local · reglas: sin LLM")
    usar_deteccion = st.checkbox("Detectar y recortar el plato (OpenCV)", value=True)
    usar_clahe = st.checkbox("Mejorar iluminación (CLAHE)", value=True)

    with st.expander("Mi perfil (meta calórica)"):
        sexo = st.radio("Sexo", ["femenino", "masculino"], horizontal=True)
        edad = st.number_input("Edad", 15, 90, 21)
        peso = st.number_input("Peso (kg)", 30.0, 200.0, 65.0)
        estatura = st.number_input("Estatura (cm)", 120.0, 220.0, 168.0)
        actividad = st.selectbox("Actividad física", ["sedentaria", "ligera", "moderada", "alta"])
        if st.button("Calcular meta"):
            diario.meta_kcal = meta_calorica(sexo, edad, peso, estatura, actividad)
    if diario.meta_kcal:
        st.metric("Meta diaria estimada", f"{diario.meta_kcal:.0f} kcal")
    st.caption("Valores nutricionales aproximados. No reemplaza la orientación de un profesional de la salud.")

try:
    clf = obtener_clasificador(modelo)
except Exception as e:
    st.error(f"No se pudo cargar el modelo '{modelo}': {e}. Revisa requirements.txt "
             "o entrena la CNN con el notebook 01 y copia el .keras en models/.")
    st.stop()
if "asistente" not in st.session_state:
    st.session_state.asistente = AsistenteNutricional(base, diario, motor)
asistente: AsistenteNutricional = st.session_state.asistente
asistente.motor = motor

tab_det, tab_diario, tab_chat = st.tabs(["📷 Detectar plato", "📒 Diario calórico", "💬 NutriBot"])

# ---------------- Detección ----------------
with tab_det:
    col1, col2 = st.columns(2)
    with col1:
        fuente = st.radio("Fuente de imagen", ["Cámara", "Subir archivo"], horizontal=True)
        archivo = st.camera_input("Toma una foto del plato") if fuente == "Cámara" else \
            st.file_uploader("Imagen del plato", type=["jpg", "jpeg", "png", "webp"])
    if archivo is not None:
        img = cv2.imdecode(np.frombuffer(archivo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        proc = mejorar_iluminacion(img) if usar_clahe else img
        h, w = img.shape[:2]
        caja = detectar_plato(proc) if usar_deteccion else (0, 0, w, h)
        top = clf.predecir(preparar_para_modelo(recortar(proc, caja)))
        clase, conf = top[0]
        asistente.ultimo_detectado = clase
        info = base[clase]
        with col2:
            st.image(cv2.cvtColor(dibujar_resultado(img, caja, base.nombre(clase), conf, info["kcal"]),
                                  cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Modelo: {clf.nombre}")
            if conf < UMBRAL_CONFIANZA:
                st.warning("Confianza baja: la foto puede ser ambigua o el plato no está entre las 101 clases.")
            st.dataframe(pd.DataFrame({"Alimento": [base.nombre(c) for c, _ in top],
                                       "Probabilidad": [f"{p:.1%}" for _, p in top]}),
                         hide_index=True, use_container_width=True)
            st.subheader(f"{info['nombre_es']} · {info['porcion']}")
            factor = st.slider("Tamaño de la porción", 0.25, 3.0, 1.0, 0.25)
            m = st.columns(4)
            m[0].metric("Calorías", f"{info['kcal'] * factor:.0f} kcal")
            m[1].metric("Proteína", f"{info['proteina_g'] * factor:.0f} g")
            m[2].metric("Carbohidratos", f"{info['carbohidratos_g'] * factor:.0f} g")
            m[3].metric("Grasa", f"{info['grasa_g'] * factor:.0f} g")
            if st.button("➕ Registrar en mi diario", type="primary"):
                diario.registrar(clase, factor)
                st.success("Registrado.")

# ---------------- Diario ----------------
with tab_diario:
    if not diario.consumos:
        st.info("Aún no has registrado alimentos hoy.")
    else:
        t = diario.totales()
        c = st.columns(4)
        c[0].metric("Total", f"{t['kcal']:.0f} kcal",
                    f"{diario.meta_kcal - t['kcal']:.0f} restantes" if diario.meta_kcal else None)
        c[1].metric("Proteína", f"{t['proteina_g']:.0f} g")
        c[2].metric("Carbohidratos", f"{t['carbohidratos_g']:.0f} g")
        c[3].metric("Grasa", f"{t['grasa_g']:.0f} g")
        if diario.meta_kcal:
            st.progress(min(t["kcal"] / diario.meta_kcal, 1.0), text="Avance hacia la meta diaria")
        st.dataframe(pd.DataFrame([{"Hora": x.hora, "Alimento": base.nombre(x.clase), "Porciones": x.factor,
                                    "kcal": round(x.kcal), "Proteína (g)": round(x.proteina_g),
                                    "Carbohidratos (g)": round(x.carbohidratos_g), "Grasa (g)": round(x.grasa_g)}
                                   for x in diario.consumos]), hide_index=True, use_container_width=True)
        kcal_macros = {"Proteína": t["proteina_g"] * 4, "Carbohidratos": t["carbohidratos_g"] * 4,
                       "Grasa": t["grasa_g"] * 9}
        st.bar_chart(pd.Series(kcal_macros, name="kcal por macronutriente"))
        if st.button("Reiniciar día"):
            diario.consumos.clear()
            st.rerun()

# ---------------- Asistente ----------------
with tab_chat:
    st.caption("Pregunta por calorías, porciones, alternativas más ligeras o tu total del día. "
               + (f"Último plato detectado: **{base.nombre(asistente.ultimo_detectado)}**."
                  if asistente.ultimo_detectado else ""))
    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).markdown(msg["content"])
    if pregunta := st.chat_input("Ej.: ¿Cuántas calorías llevo hoy? ¿Qué puedo comer más ligero que la pizza?"):
        st.chat_message("user").markdown(pregunta)
        with st.spinner("NutriBot está pensando..."):
            respuesta = asistente.responder(pregunta)
        st.chat_message("assistant").markdown(respuesta)
        st.session_state.chat += [{"role": "user", "content": pregunta},
                                  {"role": "assistant", "content": respuesta}]
