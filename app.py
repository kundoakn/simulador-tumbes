import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Simulador - Cuenca del Tumbes",
    page_icon="🌧️",
    layout="wide"
)

# ============================================
# ESTILO PERSONALIZADO
# ============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header h3 {
        color: #e3f2fd;
        margin: 5px 0 0 0;
    }
    .metric-box {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# TÍTULO
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🌧️ Simulador de Predicción Pluviométrica</h1>
    <h3>Cuenca del Río Tumbes - Perú</h3>
</div>
""", unsafe_allow_html=True)

# ============================================
# INICIALIZAR VARIABLES DE SESIÓN
# ============================================
if 'datos_generados' not in st.session_state:
    st.session_state.datos_generados = False
if 'df_real' not in st.session_state:
    st.session_state.df_real = None
if 'df_final' not in st.session_state:
    st.session_state.df_final = None
if 'df_imputado' not in st.session_state:
    st.session_state.df_imputado = None
if 'resultados' not in st.session_state:
    st.session_state.resultados = {}

# ============================================
# FUNCIONES
# ============================================
def generar_datos(porcentaje):
    """Genera datos sintéticos con patrón climático de Tumbes"""
    np.random.seed(42)
    dias = 730
    fecha = pd.date_range(start='2022-01-01', periods=dias, freq='D')
    
    # Patrón climático de Tumbes
    dia_del_año = np.array([(f - fecha[0]).days % 365 for f in fecha])
    t = 2 * np.pi * (dia_del_año - 80) / 365
    
    # Estaciones
    base_rain = 55 + 50 * np.cos(t) + 12 * np.sin(2*t) + np.random.normal(0, 6, dias)
    base_rain = np.maximum(base_rain, 0)
    base_rain = np.round(base_rain, 1)
    
    zorritos = 0.65 * base_rain + 8 * np.sin(t + 0.5) + np.random.normal(0, 4, dias)
    zorritos = np.maximum(zorritos, 0)
    zorritos = np.round(zorritos, 1)
    
    pampas = 1.35 * base_rain + 15 * np.cos(t/2 + 0.3) + np.random.normal(0, 9, dias)
    pampas = np.maximum(pampas, 0)
    pampas = np.round(pampas, 1)
    
    df_real = pd.DataFrame({
        'Fecha': fecha,
        'Tumbes': base_rain,
        'Zorritos': zorritos,
        'Pampas': pampas
    })
    
    # Inyectar datos faltantes
    pct = porcentaje / 100.0
    mask = np.random.random(df_real[['Tumbes', 'Zorritos', 'Pampas']].shape) < pct
    
    df_final = df_real[['Fecha']].copy()
    for col in ['Tumbes', 'Zorritos', 'Pampas']:
        df_final[col] = df_real[col].copy()
        df_final.loc[mask[:, ['Tumbes', 'Zorritos', 'Pampas'].index(col)], col] = np.nan
    
    st.session_state.df_real = df_real
    st.session_state.df_final = df_final
    st.session_state.df_imputado = df_final.copy()
    st.session_state.datos_generados = True
    st.session_state.resultados = {}

def ejecutar_modelo(modelo):
    """Ejecuta el modelo seleccionado"""
    if not st.session_state.datos_generados:
        st.warning("⚠️ Primero genera los datos")
        return
    
    df_final = st.session_state.df_final
    df_real = st.session_state.df_real
    df_imputado = df_final.copy()
    resultados_modelo = {}
    
    for estacion_objetivo in ['Tumbes', 'Zorritos', 'Pampas']:
        columnas_predictoras = [col for col in ['Tumbes', 'Zorritos', 'Pampas'] if col != estacion_objetivo]
        
        datos_conocidos = df_final[df_final[estacion_objetivo].notna()]
        datos_limpios = datos_conocidos[columnas_predictoras + [estacion_objetivo]].dropna()
        
        if len(datos_limpios) < 2:
            continue
        
        X_train = datos_limpios[columnas_predictoras]
        y_train = datos_limpios[estacion_objetivo]
        
        datos_faltantes = df_final[df_final[estacion_objetivo].isna()]
        if len(datos_faltantes) == 0:
            continue
        
        X_predict = datos_faltantes[columnas_predictoras]
        X_predict_limpio = X_predict.dropna()
        
        if len(X_predict_limpio) == 0:
            continue
        
        indices_predecir = X_predict_limpio.index
        
        if modelo == 'lineal':
            model = LinearRegression()
            nombre = "Regresión Lineal"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            nombre = "Random Forest"
        
        model.fit(X_train, y_train)
        pred = model.predict(X_predict_limpio)
        
        df_imputado.loc[indices_predecir, f'{estacion_objetivo}_pred'] = np.round(pred, 1)
        
        if len(indices_predecir) > 0:
            y_real = df_real.loc[indices_predecir, estacion_objetivo]
            rmse = np.sqrt(mean_squared_error(y_real, pred))
            r2 = r2_score(y_real, pred)
            
            resultados_modelo[estacion_objetivo] = {
                'RMSE': rmse,
                'R²': r2,
                'Modelo': nombre,
                'Predichos': len(indices_predecir)
            }
    
    st.session_state.df_imputado = df_imputado
    st.session_state.resultados[modelo] = resultados_modelo

# ============================================
# BARRA LATERAL
# ============================================
with st.sidebar:
    st.markdown("## 🎛️ Panel de Control")
    
    # Generar datos
    st.markdown("### 📊 Generar Datos")
    porcentaje = st.slider(
        "% Datos Faltantes",
        min_value=5,
        max_value=40,
        value=15,
        step=5
    )
    
    if st.button("🔄 Generar Nuevos Datos", use_container_width=True):
        with st.spinner("Generando datos..."):
            generar_datos(porcentaje)
        st.success("✅ Datos generados correctamente")
        st.rerun()
    
    st.divider()
    
    # Modelos
    st.markdown("### 🤖 Modelos de IA")
    
    if st.button("📊 Regresión Lineal", use_container_width=True):
        with st.spinner("Ejecutando..."):
            ejecutar_modelo('lineal')
        st.success("✅ Listo")
        st.rerun()
    
    if st.button("🌲 Random Forest", use_container_width=True):
        with st.spinner("Ejecutando..."):
            ejecutar_modelo('random_forest')
        st.success("✅ Listo")
        st.rerun()
    
    if st.button("🚀 Ejecutar Ambos", use_container_width=True, type="primary"):
        with st.spinner("Ejecutando..."):
            ejecutar_modelo('lineal')
            ejecutar_modelo('random_forest')
        st.success("✅ Ambos ejecutados")
        st.rerun()
    
    st.divider()
    
    # Información
    st.markdown("### 📋 Info Climática")
    st.info("""
    **🌧️ Lluvioso**  
    Dic - May | Pico: Marzo
    
    **☀️ Seco**  
    Jun - Nov | Mínimo: Agosto
    """)

# ============================================
# ÁREA PRINCIPAL
# ============================================
if not st.session_state.datos_generados:
    st.info("👈 Usa el panel izquierdo para generar datos")
else:
    # Selector de estación
    estacion = st.selectbox(
        "📍 Seleccionar Estación",
        ['Tumbes', 'Zorritos', 'Pampas']
    )
    
    # Gráficos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    df_real = st.session_state.df_real
    df_final = st.session_state.df_final
    df_imputado = st.session_state.df_imputado
    resultados = st.session_state.resultados
    
    # Gráfico 1
    datos_reales = df_real[estacion]
    datos_faltantes = df_final[estacion]
    
    ax1.plot(df_real['Fecha'], datos_reales, 
            label='Datos Reales', color='#1a237e', alpha=0.7, linewidth=1.5)
    
    mask_nan = datos_faltantes.isna()
    if mask_nan.any():
        ax1.scatter(df_real.loc[mask_nan, 'Fecha'], 
                   df_real.loc[mask_nan, estacion],
                   color='red', s=25, label='Datos Faltantes', alpha=0.8)
    
    # Sombreado de temporadas
    for year in [2022, 2023]:
        ax1.axvspan(pd.Timestamp(f'{year}-12-01'), pd.Timestamp(f'{year+1}-05-31'), 
                   alpha=0.1, color='blue')
        ax1.axvspan(pd.Timestamp(f'{year}-06-01'), pd.Timestamp(f'{year}-11-30'), 
                   alpha=0.1, color='orange')
    
    ax1.set_title(f'🌧️ Estación {estacion} - Datos Originales', fontsize=12)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Precipitación (mm)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2
    if resultados:
        colores = {'lineal': '#2196F3', 'random_forest': '#FF9800'}
        nombres = {'lineal': 'Regresión Lineal', 'random_forest': 'Random Forest'}
        
        ax2.plot(df_real['Fecha'], datos_reales, 
                label='Datos Reales', color='green', alpha=0.2)
        
        for modelo in ['lineal', 'random_forest']:
            if modelo in resultados:
                col_pred = f'{estacion}_pred'
                if col_pred in df_imputado.columns:
                    mask_pred = df_imputado[col_pred].notna()
                    if mask_pred.any():
                        ax2.scatter(df_imputado.loc[mask_pred, 'Fecha'],
                                   df_imputado.loc[mask_pred, col_pred],
                                   label=f'{nombres[modelo]} (predicción)',
                                   color=colores[modelo], s=25, alpha=0.7)
        
        ax2.set_title(f'🎯 Predicciones - Estación {estacion}', fontsize=12)
        ax2.set_xlabel('Fecha')
        ax2.set_ylabel('Precipitación (mm)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Ejecuta un modelo para ver predicciones',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, color='gray')
        ax2.set_title('⏳ Esperando predicciones', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ============================================
    # ESTADÍSTICAS
    # ============================================
    if resultados:
        st.divider()
        st.markdown("## 📈 Resultados")
        
        for modelo, res in resultados.items():
            nombre = "📊 Lineal" if modelo == 'lineal' else "🌲 Forest"
            st.markdown(f"**{nombre}**")
            
            if res:
                df_metrics = pd.DataFrame(res).T.round({'RMSE': 2, 'R²': 3})
                st.dataframe(df_metrics, use_container_width=True)
        
        # Comparación
        if 'lineal' in resultados and 'random_forest' in resultados:
            if resultados['lineal'] and resultados['random_forest']:
                rmse_lr = np.mean([m['RMSE'] for m in resultados['lineal'].values()])
                rmse_rf = np.mean([m['RMSE'] for m in resultados['random_forest'].values()])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Lineal RMSE", f"{rmse_lr:.2f} mm")
                col2.metric("Forest RMSE", f"{rmse_rf:.2f} mm")
                col3.metric("✅ Mejor", "Random Forest" if rmse_rf < rmse_lr else "Lineal")

# ============================================
# PIE
# ============================================
st.divider()
st.caption("🌧️ Simulador - Cuenca del Río Tumbes | Datos basados en SENAMHI")