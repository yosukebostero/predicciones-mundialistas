import json
import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.stats import poisson

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="DataFootball Analytics", page_icon="⚽", layout="centered")

# --- CONSTANTES DE CONFIGURACIÓN DEL MODELO ---
VENTANA_ANALISIS = 10     # cantidad de partidos recientes que se consideran por equipo
MAX_GOLES_MODELO = 10     # tope de goles para la matriz de Poisson (0..MAX_GOLES_MODELO-1)
PESO_DECREMENTO = 0.1     # cuánto pierde de peso cada partido según se aleja del más reciente
PESO_MINIMO = 0.1         # piso para que ningún partido quede en peso cero o negativo
MAX_FILAS_HEATMAP = 6     # tamaño del heatmap que se muestra (recortado del MAX_GOLES_MODELO real)
TOP_N_MARCADORES = 5      # cuántos marcadores mostrar en el ranking


# --- CAPA DE DATOS ---

@st.cache_data
def cargar_datos_crudos(ruta_archivo="partidos_2026.csv"):
    df = pd.read_csv(ruta_archivo)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df.sort_values(by='fecha', ascending=False)


def obtener_historial_equipo(df, equipo):
    """Filtra el historial de un equipo específico.

    OJO: a propósito NO está cacheada. @st.cache_data tiene que
    hashear el DataFrame completo en cada llamada, y con datasets
    grandes ese hasheo puede terminar costando más que el propio
    filtrado. Lo único que vale la pena cachear es la carga desde
    disco (cargar_datos_crudos) y listados derivados chicos, como
    obtener_selecciones más abajo.
    """
    df_local = df[df['local'] == equipo].copy()
    df_local['goles_favor'] = df_local['goles_local']
    df_local['goles_contra'] = df_local['goles_visitante']

    df_visita = df[df['visitante'] == equipo].copy()
    df_visita['goles_favor'] = df_visita['goles_visitante']
    df_visita['goles_contra'] = df_visita['goles_local']

    return pd.concat([df_local, df_visita]).sort_values(by='fecha', ascending=False)


@st.cache_data
def obtener_selecciones(df):
    """Lista ordenada y sin duplicados de equipos que aparecen como local o visitante."""
    return sorted(set(df['local']).union(df['visitante']))


def calcular_factor_localia(df):
    """Ventaja de jugar en casa, estimada con todo el histórico del torneo.

    Es la razón entre el promedio de goles marcados como local y el
    promedio de goles marcados como visitante. > 1 significa que,
    en promedio, jugar de local suma goles extra.
    """
    promedio_local = df['goles_local'].mean()
    promedio_visitante = df['goles_visitante'].mean()
    if promedio_visitante == 0:
        return 1.0
    return promedio_local / promedio_visitante


# --- UTILIDADES DEL MODELO ---

def calcular_pesos_recientes(n_partidos):
    """Pesos decrecientes por antigüedad para el promedio ofensivo/defensivo.

    El partido más reciente pesa 1.0, cada uno anterior resta
    PESO_DECREMENTO, con un piso en PESO_MINIMO para que ningún
    partido termine con peso cero o negativo. Asume que los partidos
    vienen ordenados de más reciente a más antiguo (índice 0 = último jugado).
    """
    pesos = 1.0 - PESO_DECREMENTO * np.arange(n_partidos)
    return np.clip(pesos, PESO_MINIMO, 1.0)


def obtener_top_marcadores(matriz_prob, total_prob, top_n=TOP_N_MARCADORES):
    """Devuelve los top_n marcadores más probables como lista de
    (goles_local, goles_visita, probabilidad_normalizada_en_%).
    Reutiliza el mismo total_prob usado para normalizar el resto de
    las probabilidades, para que todo quede consistente.
    """
    indices_planos = np.argsort(matriz_prob, axis=None)[::-1][:top_n]
    filas, columnas = np.unravel_index(indices_planos, matriz_prob.shape)
    return [
        (int(f), int(c), float(matriz_prob[f, c] / total_prob * 100))
        for f, c in zip(filas, columnas)
    ]


# --- CAPA DE LÓGICA (pura, sin Streamlit adentro) ---

@dataclass
class EstadisticasEquipo:
    promedio_favor: float
    promedio_contra: float
    racha: str
    victorias_consec: int
    derrotas_consec: int
    partidos: int


class AnalizadorEquipo:
    def __init__(self, nombre, historial, promedio_global_torneo):
        self.nombre = nombre
        self.historial = historial
        self.promedio_global = promedio_global_torneo
        self.stats = self._calcular_estadisticas()
        self.fuerzas = self._calcular_fuerzas_base()

    def _calcular_estadisticas(self, ventana=VENTANA_ANALISIS):
        muestra = self.historial.head(ventana).copy()

        if muestra.empty:
            return EstadisticasEquipo(
                promedio_favor=0.0, promedio_contra=0.0, racha="N/A",
                victorias_consec=0, derrotas_consec=0, partidos=0
            )

        # Vectorizado con np.where en vez de apply(axis=1)
        muestra['resultado'] = np.where(
            muestra['goles_favor'] > muestra['goles_contra'], 'W',
            np.where(muestra['goles_favor'] < muestra['goles_contra'], 'L', 'D')
        )

        resultados = muestra['resultado'].tolist()
        v, d = 0, 0
        # Criterio elegido: un empate corta tanto la racha de victorias
        # como la de derrotas (no se considera "invicto"). Es una
        # decisión de diseño válida entre varias posibles, documentada
        # aquí para que quede explícita.
        for res in resultados:
            if res == 'W':
                v += 1
                d = 0 if d > 0 else d
            elif res == 'L':
                d += 1
                v = 0 if v > 0 else v
            else:
                break

        pesos = calcular_pesos_recientes(len(muestra))
        promedio_favor = np.average(muestra['goles_favor'], weights=pesos)
        promedio_contra = np.average(muestra['goles_contra'], weights=pesos)

        return EstadisticasEquipo(
            promedio_favor=promedio_favor,
            promedio_contra=promedio_contra,
            racha=" - ".join(resultados[::-1]),
            victorias_consec=v,
            derrotas_consec=d,
            partidos=len(muestra)
        )

    def _calcular_fuerzas_base(self):
        """Índices relativos de ataque/defensa frente al promedio del torneo.

        Esto NO son lambdas todavía, son solo índices de fuerza.
        Las lambdas reales del modelo de Poisson se arman después,
        combinando ataque local x defensa visitante x promedio global
        (y opcionalmente el factor de localía).
        """
        ataque = self.stats.promedio_favor / self.promedio_global if self.promedio_global > 0 else 0
        defensa = self.stats.promedio_contra / self.promedio_global if self.promedio_global > 0 else 0
        return {"ataque": ataque, "defensa": defensa}


# --- CAPA DE PRESENTACIÓN (separada a propósito de AnalizadorEquipo) ---

def renderizar_alertas(analizador):
    """Recibe un AnalizadorEquipo ya calculado y solo se encarga de pintarlo.

    Al vivir fuera de la clase, el motor de cálculo (AnalizadorEquipo)
    queda libre de cualquier dependencia de Streamlit y se podría
    reutilizar tal cual en Flask, FastAPI, un notebook, etc.
    """
    stats = analizador.stats
    st.write(f"**Últimos resultados:** `{stats.racha}`")
    if stats.derrotas_consec >= 3:
        st.error(f"🚑 ¡Alerta! **{analizador.nombre}** acumula {stats.derrotas_consec} derrotas al hilo. "
                 f"Ni presentándose ganan hoy.")
    elif stats.victorias_consec >= 3:
        st.success(f"🔥 ¡Imparables! **{analizador.nombre}** lleva {stats.victorias_consec} victorias seguidas.")
    elif stats.partidos < 5:
        st.warning(f"⚠️ Pocos datos históricos para {analizador.nombre}.")
    else:
        st.info("📊 Rendimiento estable dentro de la media.")


def renderizar_metricas(analizador):
    """Tarjetas de métricas (st.metric) con los números clave del equipo."""
    stats = analizador.stats
    fuerzas = analizador.fuerzas
    m1, m2 = st.columns(2)
    m1.metric("⚽ Prom. goles", f"{stats.promedio_favor:.2f}")
    m2.metric("🛡 Prom. recibidos", f"{stats.promedio_contra:.2f}")
    m3, m4 = st.columns(2)
    m3.metric("🔥 Índice ofensivo", f"{fuerzas['ataque']:.2f}")
    m4.metric("🧱 Índice defensivo", f"{fuerzas['defensa']:.2f}")


def renderizar_heatmap(matriz_prob, equipo_local, equipo_visita):
    """Heatmap de la matriz de Poisson: filas = goles del local, columnas = goles del visitante.

    Se recorta a MAX_FILAS_HEATMAP para que la tabla siga siendo legible;
    los marcadores más extremos casi no aportan probabilidad de todos modos.
    """
    n = min(MAX_FILAS_HEATMAP, matriz_prob.shape[0])
    sub_matriz = matriz_prob[:n, :n] * 100
    df_heatmap = pd.DataFrame(
        sub_matriz,
        index=[f"{equipo_local} {i}" for i in range(n)],
        columns=[f"{equipo_visita} {j}" for j in range(n)]
    )
    st.write("#### 🔥 Mapa de calor de marcadores")
    try:
        st.dataframe(
            df_heatmap.style.background_gradient(cmap="Blues", axis=None).format("{:.1f}%"),
            use_container_width=True
        )
    except ImportError:
        # background_gradient necesita matplotlib instalado; si no está,
        # se muestra la tabla sin colorear en lugar de romper la app.
        st.caption("Instala `matplotlib` (pip install matplotlib) para ver la tabla coloreada.")
        st.dataframe(df_heatmap.style.format("{:.1f}%"), use_container_width=True)


def renderizar_ranking_marcadores(top_marcadores, equipo_local, equipo_visita):
    """Tabla con los N marcadores más probables, ordenados de mayor a menor."""
    st.write(f"#### 🏆 Top {len(top_marcadores)} marcadores más probables")
    df_top = pd.DataFrame([
        {"Marcador": f"{equipo_local} {gl} - {gv} {equipo_visita}", "Probabilidad": f"{p:.1f}%"}
        for gl, gv, p in top_marcadores
    ])
    st.table(df_top)


# --- INTERFAZ PRINCIPAL ---

st.title("⚽ DataFootball Analytics Engine")
st.subheader("Modelo predictivo basado en la Distribución de Poisson")

try:
    df_partidos = cargar_datos_crudos()

    # Manejo de división por cero: CSV vacío
    if df_partidos.empty:
        st.warning("⚠️ El archivo `partidos.csv` no tiene partidos cargados. Agrega datos para continuar.")
        st.stop()

    total_goles = df_partidos['goles_local'].sum() + df_partidos['goles_visitante'].sum()
    promedio_goles_global = total_goles / (len(df_partidos) * 2)
    factor_localia = calcular_factor_localia(df_partidos)

    selecciones = obtener_selecciones(df_partidos)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        equipo_local = st.selectbox("🏟️ Selección Local", selecciones, index=0)
    with col2:
        equipo_visita = st.selectbox("✈️ Selección Visitante", [e for e in selecciones if e != equipo_local], index=0)

    usar_factor_localia = st.checkbox(
        f"Aplicar factor de localía calculado ({factor_localia:.2f}x)",
        value=True,
        help="Multiplica la lambda del local por la ventaja histórica de jugar en casa."
    )

    # Instanciación de objetos (capa de cálculo, sin Streamlit adentro)
    local = AnalizadorEquipo(equipo_local, obtener_historial_equipo(df_partidos, equipo_local), promedio_goles_global)
    visita = AnalizadorEquipo(equipo_visita, obtener_historial_equipo(df_partidos, equipo_visita), promedio_goles_global)

    st.markdown("### 📈 Análisis de Forma Reciente")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"**{equipo_local}**")
        renderizar_alertas(local)
        renderizar_metricas(local)
    with col4:
        st.markdown(f"**{equipo_visita}**")
        renderizar_alertas(visita)
        renderizar_metricas(visita)

    st.markdown("---")

    if st.button("🧮 Calcular Estimaciones del Modelo", use_container_width=True):

        # Aquí sí nacen las lambdas de verdad
        lambda_local = local.fuerzas["ataque"] * visita.fuerzas["defensa"] * promedio_goles_global
        lambda_visita = visita.fuerzas["ataque"] * local.fuerzas["defensa"] * promedio_goles_global

        if usar_factor_localia:
            lambda_local *= factor_localia

        # Matriz de probabilidades sin doble for: producto externo de dos Poisson 1D
        prob_goles_local = poisson.pmf(np.arange(MAX_GOLES_MODELO), lambda_local)
        prob_goles_visita = poisson.pmf(np.arange(MAX_GOLES_MODELO), lambda_visita)
        matriz_prob = np.outer(prob_goles_local, prob_goles_visita)

        # Sumas brutas (fila = goles local, columna = goles visitante)
        p_empate_bruto = np.sum(np.diag(matriz_prob))
        p_local_bruto = np.sum(np.tril(matriz_prob, -1))   # fila > columna -> gana local
        p_visita_bruto = np.sum(np.triu(matriz_prob, 1))   # columna > fila -> gana visita

        # Normalización para que sumen 1.0 exacto
        total_prob = p_empate_bruto + p_local_bruto + p_visita_bruto
        p_empate = (p_empate_bruto / total_prob) * 100
        p_local = (p_local_bruto / total_prob) * 100
        p_visita = (p_visita_bruto / total_prob) * 100

        top_marcadores = obtener_top_marcadores(matriz_prob, total_prob, TOP_N_MARCADORES)
        gl_optimo, gv_optimo, _ = top_marcadores[0]

        st.success(f"### 🎯 Marcador estimado óptimo: **{equipo_local} {gl_optimo} - {gv_optimo} {equipo_visita}**")

        st.write("#### Probabilidades estimadas por el modelo:")

        st.write(f"**Victoria {equipo_local}:** {p_local:.1f}%")
        st.progress(min(p_local / 100, 1.0))

        st.write(f"**Empate:** {p_empate:.1f}%")
        st.progress(min(p_empate / 100, 1.0))

        st.write(f"**Victoria {equipo_visita}:** {p_visita:.1f}%")
        st.progress(min(p_visita / 100, 1.0))

        st.markdown("---")
        renderizar_heatmap(matriz_prob, equipo_local, equipo_visita)

        st.markdown("---")
        renderizar_ranking_marcadores(top_marcadores, equipo_local, equipo_visita)

        st.markdown("---")
        st.write("#### 📤 Exportar resultados")

        resultado_exportable = {
            "equipo_local": equipo_local,
            "equipo_visita": equipo_visita,
            "lambda_local": round(float(lambda_local), 4),
            "lambda_visita": round(float(lambda_visita), 4),
            "factor_localia_aplicado": usar_factor_localia,
            "probabilidad_victoria_local_pct": round(p_local, 2),
            "probabilidad_empate_pct": round(p_empate, 2),
            "probabilidad_victoria_visita_pct": round(p_visita, 2),
            "marcador_optimo": f"{gl_optimo}-{gv_optimo}",
            "top_marcadores": [
                {"marcador": f"{gl}-{gv}", "probabilidad_pct": round(p, 2)}
                for gl, gv, p in top_marcadores
            ],
            "racha_local": local.stats.racha,
            "racha_visita": visita.stats.racha,
        }

        col_export1, col_export2 = st.columns(2)
        with col_export1:
            st.download_button(
                "⬇️ Descargar JSON",
                data=json.dumps(resultado_exportable, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"prediccion_{equipo_local}_vs_{equipo_visita}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_export2:
            df_export = pd.DataFrame([{
                k: v for k, v in resultado_exportable.items() if k != "top_marcadores"
            }])
            st.download_button(
                "⬇️ Descargar CSV",
                data=df_export.to_csv(index=False).encode("utf-8"),
                file_name=f"prediccion_{equipo_local}_vs_{equipo_visita}.csv",
                mime="text/csv",
                use_container_width=True,
            )

except FileNotFoundError:
    st.error("🚨 Archivo `partidos.csv` no encontrado. Súbelo al repositorio.")