# ============================================================
# APLICACIÓN STREAMLIT – MODELO HÍBRIDO RNA + CRITERIO ESTRUCTURAL
# MODELO FINAL (5 ENTRADAS)
# ============================================================

import os
import numpy as np
import joblib
import streamlit as st
from tensorflow.keras.models import load_model

# ------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------
st.set_page_config(
    page_title="Prediseño estructural de aulas escolares",
    layout="centered"
)

BASE_PATH = os.path.dirname(__file__)

# ------------------------------------------------------------
# CARGA DEL MODELO Y ESCALADORES
# ------------------------------------------------------------
@st.cache_resource
def cargar_modelo_final():

    model = load_model(
        os.path.join(BASE_PATH, "final_modelo_rna_estructural_3SZ.keras")
    )
    scaler_X = joblib.load(
        os.path.join(BASE_PATH, "final_scaler_X_3SZ.pkl")
    )
    scaler_Y = joblib.load(
        os.path.join(BASE_PATH, "final_scaler_Y_3SZ.pkl")
    )

    return model, scaler_X, scaler_Y


model, scaler_X, scaler_Y = cargar_modelo_final()
st.success("Modelo final cargado correctamente")

# ------------------------------------------------------------
# MAPA DE IMÁGENES POR ZONA Y CONFIGURACIÓN
# ------------------------------------------------------------
IMAGENES = {

    "Z4-S1": {
        "2 aulas por piso": {2: "planta_2a_2p.jpg", 3: "planta_2a_3p.jpg"},
        "3 aulas por piso": {2: "planta_3a_2p.jpg", 3: "planta_3a_3p.jpg"},
    },

    "Z4-S2": {
        "2 aulas por piso": {2: "planta_2a_2p.jpg", 3: "planta_2a_3p.jpg"},
        "3 aulas por piso": {2: "planta_3a_2p.jpg", 3: "planta_3a_3p.jpg"},
    },

    "Z4-S3": {
        "2 aulas por piso": {2: "planta_2a_2p.jpg", 3: "planta_2a_3p.jpg"},
        "3 aulas por piso": {2: "planta_3a_2p.jpg", 3: "planta_3a_3p.jpg"},
    },

    "Z3-S2": {
        "2 aulas por piso": {2: "planta_2a_2p.jpg", 3: "planta_2a_3p.jpg"},
        "3 aulas por piso": {2: "planta_3a_2p.jpg", 3: "planta_3a_3p.jpg"},
    },

    "Z3-S3": {
        "2 aulas por piso": {2: "planta_2a_2p.jpg", 3: "planta_2a_3p.jpg"},
        "3 aulas por piso": {2: "planta_3a_2p.jpg", 3: "planta_3a_3p.jpg"},
    },

    # 👇 AQUÍ DIFERENCIAS Z2
    "Z2-S1": {
        "2 aulas por piso": {2: "planta_2a_2p_Z2.jpg", 3: "planta_2a_3p_Z2.jpg"},
        "3 aulas por piso": {2: "planta_3a_2p_Z2.jpg", 3: "planta_3a_3p_Z2.jpg"},
    },

    "Z2-S2": {
        "2 aulas por piso": {2: "planta_2a_2p.jpg", 3: "planta_2a_3p.jpg"},
        "3 aulas por piso": {2: "planta_3a_2p.jpg", 3: "planta_3a_3p.jpg"},
    },

    "Z2-S3": {
        "2 aulas por piso": {2: "planta_2a_2p.jpg", 3: "planta_2a_3p.jpg"},
        "3 aulas por piso": {2: "planta_3a_2p.jpg", 3: "planta_3a_3p.jpg"},
    },
}

# ------------------------------------------------------------
# FACTORES SÍSMICOS (S, Z, S×Z)
# ------------------------------------------------------------
FACTORES_SZ = {
    "Z2-S1": {"S": 1.00, "Z": 0.25, "SZ": 0.250},
    "Z2-S2": {"S": 1.20, "Z": 0.25, "SZ": 0.300},
    "Z2-S3": {"S": 1.40, "Z": 0.25, "SZ": 0.350},
    "Z3-S2": {"S": 1.15, "Z": 0.35, "SZ": 0.403},
    "Z3-S3": {"S": 1.20, "Z": 0.35, "SZ": 0.420},
    "Z4-S1": {"S": 1.00, "Z": 0.45, "SZ": 0.450},
    "Z4-S2": {"S": 1.05, "Z": 0.45, "SZ": 0.473},
    "Z4-S3": {"S": 1.10, "Z": 0.45, "SZ": 0.495},
}

# ------------------------------------------------------------
# TÍTULO Y DESCRIPCIÓN
# ------------------------------------------------------------
st.title("Aplicación para el prediseño estructural de aulas escolares")

st.markdown("""
Aplicación de apoyo al prediseño estructural preliminar de aulas escolares.

La herramienta permite estimar la longitud total del sistema de placas estructurales (Ltotal)
a partir de variables geométricas y de la condición sísmica de la edificación,
proporcionando dimensiones preliminares de los principales elementos estructurales
como referencia en etapas iniciales de diseño.
""")


# ------------------------------------------------------------
# CONDICIÓN SÍSMICA
# ------------------------------------------------------------
st.subheader("Condición sísmica de la edificación (Norma E.030)")

col0, col1, col2, col3, col4 = st.columns([2.5, 1.5, 1.25, 1.25, 1.25])

with col0:
    st.write("Zona sísmica – Tipo de suelo")

with col1:
    opcion = st.selectbox(
        "",
        [
            "Z4 – S3", "Z4 – S2", "Z4 – S1",
            "Z3 – S3", "Z3 – S2",
            "Z2 – S3", "Z2 – S2", "Z2 – S1"
        ],
        label_visibility="collapsed"
    )

clave_sz = opcion.replace(" ", "").replace("–", "-")

if clave_sz not in FACTORES_SZ:
    st.error("Combinación no disponible.")
    st.stop()

datos_sz = FACTORES_SZ[clave_sz]
zona_sismica = clave_sz.split("-")[0]
Factor_SZ = datos_sz["SZ"]

with col2:
    st.write(f"**S:** {datos_sz['S']:.2f}")

with col3:
    st.write(f"**Z:** {datos_sz['Z']:.2f}")

with col4:
    st.write(f"**S × Z:** {datos_sz['SZ']:.3f}")

# ------------------------------------------------------------
# DATOS GEOMÉTRICOS
# ------------------------------------------------------------
st.subheader("Datos geométricos de entrada")

col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

with col1:
    Npisos = st.selectbox("N° pisos", [2, 3])

with col3:
    tipo_config = st.selectbox(
        "Configuración",
        ["2 aulas por piso", "3 aulas por piso"]
    )

# Dominio dinámico de Ledif según configuración
if tipo_config == "2 aulas por piso":
    Ledif_min, Ledif_max, Ledif_def = 15.83, 19.03, 19.03
else:
    Ledif_min, Ledif_max, Ledif_def = 23.63, 28.43, 23.63

with col2:
    Hnpt = st.number_input(
        "Hnpt (m)",
        min_value=3.00,
        max_value=4.20,
        value=3.00,
        step=0.05
    )

with col4:
    Bedif = st.number_input(
        "Bedif (m)",
        min_value=9.74,
        max_value=10.74,
        value=9.74,
        step=0.05
    )

with col5:
    Ledif = st.number_input(
        "Ledif (m)",
        min_value=Ledif_min,
        max_value=Ledif_max,
        value=Ledif_def,
        step=0.10
    )

    

# Dominio de Ledif según configuración
if tipo_config == "2 aulas por piso":
    Ledif_min, Ledif_max, Ledif_def = 15.83, 19.03, 19.03
else:
    Ledif_min, Ledif_max, Ledif_def = 23.63, 28.43, 23.63



st.caption("Dominio de validez del modelo predictivo:")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("Altura de entrepiso (Hnpt)")
    st.caption("3.00 m ≤ Hnpt ≤ 4.20 m")

with col2:
    st.caption("Ancho de edificación (Bedif)")
    st.caption("9.74 m ≤ Bedif ≤ 10.74 m")

with col3:
    st.caption("Largo de edificación (Ledif)")
    st.caption(f"{Ledif_min:.2f} m ≤ Ledif ≤ {Ledif_max:.2f} m")


# ------------------------------------------------------------
# CRITERIOS ESTRUCTURALES DEFINITIVOS (VERIFICADOS)
# ------------------------------------------------------------

TOL = 1e-6

# ---------------- COLUMNAS L ----------------
def columnas_L(Factor_SZ, Npisos):

    if abs(Factor_SZ - 0.25) < TOL:
        if Npisos == 2:
            return 0.25, 0.40
        else:
            return 0.40, 0.40

    return 0.40, 0.40


# ---------------- COLUMNAS T ----------------
def columnas_T(Factor_SZ, Npisos):

    if abs(Factor_SZ - 0.25) < TOL:
        return 0.30, 0.40

    if Npisos == 2:
        return 0.55, 0.40

    return 0.00, 0.00


# ---------------- VIGAS X ----------------
def vigas_X(Npisos):

    Vx_b = 0.25
    Vx_h = 0.40 if Npisos == 2 else 0.45

    return Vx_b, Vx_h


# ---------------- VIGAS Y ----------------
def viga_Y_entrepiso(Bedif, Ledif):

    b = 0.30
    EPS = 1e-6

    lim1 = (9.74 + 10.24) / 2
    lim2 = (10.24 + 10.74) / 2

    if Bedif >= lim2:
        h = 0.60
    elif Bedif >= lim1:
        h = 0.55
    else:
        if ((18.23 - EPS <= Ledif <= 19.03 + EPS) or
            (27.23 - EPS <= Ledif <= 28.43 + EPS)):
            h = 0.55
        else:
            h = 0.50

    return b, h


def viga_Y_techo(Bedif):

    b = 0.30

    lim1 = (9.74 + 10.24) / 2
    lim2 = (10.24 + 10.74) / 2

    if Bedif >= lim2:
        h = 0.60
    elif Bedif >= lim1:
        h = 0.55
    else:
        h = 0.50

    return b, h


# ---------------- PLACAS ----------------
def placas_T(Factor_SZ, Ltotal, Ledif, Bedif, Npisos):

    PLx_e = 0.25
    PLy_e = 0.30

    PLy = 0.50 if Bedif >= 10.49 else 0.40

    if abs(Factor_SZ - 0.25) < TOL:
        if Ledif <= 19.03:
            n = 4
        else:
            n = 6
    else:
        if Ledif <= 19.03:
            n = 4 if Npisos == 2 else 6
        else:
            n = 6 if Npisos == 2 else 10

    PLx = Ltotal / n

    return PLx, PLy, PLx_e, PLy_e, n


# ------------------------------------------------------------
# PREDICCIÓN
# ------------------------------------------------------------
if st.button("Ejecutar estimación y predimensionamiento"):

    X = np.array([[Hnpt, Npisos, Bedif, Ledif, Factor_SZ]])
    Xs = scaler_X.transform(X)
    Ys = model.predict(Xs, verbose=0)
    Ltotal = scaler_Y.inverse_transform(Ys)[0][0]
    
    st.subheader("Longitud total del sistema de placas estructurales (Ltotal)")

    col_texto, col_valor = st.columns([4, 1])

    with col_texto:
        st.markdown(
        "Resultado del modelo predictivo desarrollado en el estudio"
    )
    with col_valor:
        st.markdown(
            f"<div style='text-align:right; font-size:1.5rem; font-weight:600;'>{Ltotal:.3f} m</div>",
            unsafe_allow_html=True
    )

    st.subheader("Predimensionamiento estructural")

    st.info(
    "Las dimensiones corresponden a criterios de predimensionamiento "
    "estructural definidos a partir del análisis de la base de datos "
    "del estudio y constituyen una referencia preliminar para el diseño."
)


    # --- Cálculo de elementos ---
    Tx, Ty = columnas_T(Factor_SZ, Npisos)
    Lx_c, Ly_c = columnas_L(Factor_SZ, Npisos)

    Vx_b, Vx_h = vigas_X(Npisos)

    VyE_b, VyE_h = viga_Y_entrepiso(Bedif, Ledif)
    VyT_b, VyT_h = viga_Y_techo(Bedif)

    PLx, PLy, ex, ey, nplacas = placas_T(
    Factor_SZ, Ltotal, Ledif, Bedif, Npisos
)

# ------------------------------------------------------------
# CONFIGURACIÓN ESTRUCTURAL ASOCIADA AL PREDIMENSIONAMIENTO
# ------------------------------------------------------------
    try:
        img = IMAGENES[clave_sz][tipo_config][Npisos]
        st.image(
            os.path.join(BASE_PATH, img),
            use_container_width=True
        )
    except KeyError:
        st.warning("No se dispone de un esquema para esta configuración.")


    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            **Columna tipo T:**  
            - Dimensión en dirección x (Tx): {Tx:.2f} m  
            - Dimensión en dirección y (Ty): {Ty:.2f} m  

            **Columna tipo L:**  
            - Dimensión en dirección x (Lx): {Lx_c:.2f} m  
            - Dimensión en dirección y (Ly): {Ly_c:.2f} m
            """
        )

    with col2:
        st.markdown("**Vigas de concreto armado**")
        st.markdown(
            f"""
            - Dirección X: ancho b = {Vx_b:.2f} m; altura h = {Vx_h:.2f} m  
            - Dirección Y (entrepiso): b = {VyE_b:.2f} m; h = {VyE_h:.2f} m  
            - Dirección Y (techo): b = {VyT_b:.2f} m; h = {VyT_h:.2f} m
            """
        )

        st.markdown("**Placas estructurales tipo T**")
        st.markdown(
            f"""
            - Longitud en dirección x (PLx): {PLx:.2f} m  
            - Longitud en dirección y (PLy): {PLy:.2f} m  
            - Espesor de placa en dirección x (PLx-e): {ex:.2f} m  
            - Espesor de placa en dirección y (PLy-e): {ey:.2f} m  
            - Número total de placas (N): **{nplacas}**
            """
        )
        st.caption("PLx = Ltotal / N")




