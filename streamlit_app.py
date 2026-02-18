import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(
    page_title="Primeira Analytics - Scouting System",
    page_icon="⚽",
    layout="wide",
)

@st.cache_data
def load_data():
    file_id = "1Girk_-QfK4hggiOdBpmHH5fW6JzfKc03"
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

@st.cache_resource
def load_config():
    weights = joblib.load('archetype_weights.joblib')
    config = joblib.load('config.joblib')
    return weights, config

df = load_data()
weights, config = load_config()

archetype_weights = weights['archetype_weights']
POSITION_MAP = config['POSITION_TO_ARCHETYPES_NORM']
ALL_PERF_FEATURES = config['ALL_PERFORMANCE_FEATURES']

# Sidebar
st.sidebar.title("⚽ Primeira Analytics")
st.sidebar.markdown("---")

module = st.sidebar.radio(
    "Selecciona un módulo:",
    [
        "🔍 Módulo 1: Best by Budget",
        "🔄 Módulo 2: Find Replacement",
        "💎 Módulo 3: Undervalued Players",
        "⭐ Módulo 4: Wonderkids",
        "🔄 Módulo 5: Flip Opportunities",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Datos:** 22,877 jugadores\n"
    "**Temporadas:** 2017-2025\n"
    "**Ligas:** 7 europeas"
)

# MÓDULO 1
if "Módulo 1" in module:
    st.title("🔍 Best by Budget")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        budget = st.number_input("Presupuesto (M€)", 0.5, 200.0, 30.0, 5.0) * 1_000_000
    with col2:
        position = st.selectbox("Posición", ["Todas"] + list(POSITION_MAP.keys()))
    with col3:
        season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True))
    
    min_90s = st.slider("Mínimo 90 minutos", 5, 40, 15)
    
    if st.button("🔍 Buscar", type="primary"):
        df_search = df[
            (df['season'] == season) &
            (df['market_value_current'] <= budget) &
            (df['performance_score'] >= 60) &
            (df['total_90s'] >= min_90s) &
            (df['market_value_current'].notna())
        ].copy()
        
        if position != "Todas":
            df_search = df_search[df_search['archetype'].isin(POSITION_MAP[position])]
        
        if len(df_search) == 0:
            st.warning("Sin resultados")
        else:
            df_result = df_search.nlargest(20, 'performance_score')
            
            df_display = df_result[[
                'Player', 'archetype', 'domestic_league', 'Age',
                'performance_score', 'market_value_current',
                'predicted_transfer_fee', 'gap_ratio'
            ]].copy()
            
            df_display.columns = ['Jugador', 'Arquetipo', 'Liga', 'Edad', 'Score', 'MV (TM)', 'Transfer Fee', 'Gap']
            df_display['MV (TM)'] = df_display['MV (TM)'].apply(lambda x: f"€{x/1e6:.1f}M")
            df_display['Transfer Fee'] = df_display['Transfer Fee'].apply(lambda x: f"€{x/1e6:.1f}M" if pd.notna(x) else "-")
            df_display['Gap'] = df_display['Gap'].apply(lambda x: f"x{x:.2f}" if pd.notna(x) else "-")
            df_display['Score'] = df_display['Score'].round(1)
            
            st.dataframe(df_display, use_container_width=True, height=600)

# MÓDULO 2
elif "Módulo 2" in module:
    st.title("🔄 Find Replacement")
    
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Nombre del jugador", placeholder="Ej: Pedri")
    with col2:
        budget = st.number_input("Presupuesto (M€) - opcional", 0.0, 200.0, 0.0, 5.0)
    
    season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True), key="r_season")
    
    if st.button("🔍 Buscar", type="primary"):
        if not player_name:
            st.warning("Ingresa un nombre")
        else:
            ref_candidates = df[df['Player'].str.contains(player_name, case=False, na=False)]
            if len(ref_candidates) == 0:
                st.error(f"Jugador '{player_name}' no encontrado")
            else:
                ref = ref_candidates[ref_candidates['season'] == season].iloc[0] if len(ref_candidates[ref_candidates['season'] == season]) > 0 else ref_candidates.iloc[0]
                
                st.success(f"Referencia: {ref['Player']} - {ref['archetype']}")
                
                pct_cols = [f'pct_{f}' for f in ALL_PERF_FEATURES if f'pct_{f}' in df.columns]
                arch_weights = archetype_weights.get(ref['archetype'], {})
                top_features = [f'pct_{f}' for f, _ in sorted(arch_weights.items(), key=lambda x: x[1], reverse=True)[:15] if f'pct_{f}' in df.columns]
                
                if not top_features:
                    st.error("Sin features")
                else:
                    ref_vector = ref[top_features].fillna(50).values.reshape(1, -1)
                    
                    df_pool = df[
                        (df['season'] == season) &
                        (df['archetype'] == ref['archetype']) &
                        (~df['Player'].str.contains(player_name, case=False, na=False)) &
                        (df['total_90s'] >= 10)
                    ].copy()
                    
                    if budget > 0:
                        df_pool = df_pool[df_pool['market_value_current'] <= budget * 1e6]
                    
                    if len(df_pool) == 0:
                        st.warning("Sin alternativas")
                    else:
                        pool_vectors = df_pool[top_features].fillna(50).values
                        
                        feature_weights = np.array([arch_weights.get(f.replace('pct_', ''), 1/len(top_features)) for f in top_features])
                        feature_weights = feature_weights / feature_weights.sum()
                        
                        diff = pool_vectors - ref_vector
                        distances = np.sqrt(np.sum((diff ** 2) * feature_weights, axis=1))
                        similarity = ((1 - distances / distances.max()) * 100).round(1)
                        
                        df_pool['similarity'] = similarity
                        df_result = df_pool.nlargest(15, 'similarity')
                        
                        df_display = df_result[['Player', 'domestic_league', 'Age', 'similarity', 'performance_score', 'market_value_current', 'gap_ratio']].copy()
                        df_display.columns = ['Jugador', 'Liga', 'Edad', 'Similitud', 'Score', 'MV (TM)', 'Gap']
                        df_display['Similitud'] = df_display['Similitud'].apply(lambda x: f"{x:.1f}%")
                        df_display['MV (TM)'] = df_display['MV (TM)'].apply(lambda x: f"€{x/1e6:.1f}M")
                        df_display['Gap'] = df_display['Gap'].apply(lambda x: f"x{x:.2f}" if pd.notna(x) else "-")
                        df_display['Score'] = df_display['Score'].round(1)
                        
                        st.dataframe(df_display, use_container_width=True)

# MÓDULO 3
elif "Módulo 3" in module:
    st.title("💎 Undervalued Players")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        position = st.selectbox("Posición", ["Todas"] + list(POSITION_MAP.keys()), key="u_pos")
    with col2:
        min_gap = st.slider("Gap mínimo", 1.0, 5.0, 1.5, 0.1)
    with col3:
        season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True), key="u_season")
    
    if st.button("💎 Buscar", type="primary"):
        df_search = df[
            (df['season'] == season) &
            (df['gap_ratio'] >= min_gap) &
            (df['performance_score'] >= 60) &
            (df['total_90s'] >= 15)
        ].copy()
        
        if position != "Todas":
            df_search = df_search[df_search['archetype'].isin(POSITION_MAP[position])]
        
        if len(df_search) > 0:
            df_result = df_search.nlargest(20, 'gap_ratio')
            df_display = df_result[['Player', 'archetype', 'domestic_league', 'Age', 'performance_score', 'market_value_current', 'predicted_market_value', 'gap_ratio']].copy()
            df_display.columns = ['Jugador', 'Arquetipo', 'Liga', 'Edad', 'Score', 'MV (TM)', 'MV (Pred)', 'Gap']
            for col in ['MV (TM)', 'MV (Pred)']:
                df_display[col] = df_display[col].apply(lambda x: f"€{x/1e6:.1f}M")
            df_display['Gap'] = df_display['Gap'].apply(lambda x: f"x{x:.2f}")
            df_display['Score'] = df_display['Score'].round(1)
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("Sin resultados")

# MÓDULO 4
elif "Módulo 4" in module:
    st.title("⭐ Wonderkids")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_age = st.slider("Edad máxima", 16, 23, 21)
    with col2:
        budget = st.number_input("Presupuesto (M€)", 0.0, 200.0, 50.0, 5.0, key="w_budget") * 1_000_000
    with col3:
        position = st.selectbox("Posición", ["Todas"] + list(POSITION_MAP.keys()), key="w_pos")
    
    season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True), key="w_season")
    
    if st.button("⭐ Buscar", type="primary"):
        df_search = df[
            (df['season'] == season) &
            (df['Age'] <= max_age) &
            (df['market_value_current'] <= budget) &
            (df['performance_score'] >= 50) &
            (df['total_90s'] >= 10)
        ].copy()
        
        if position != "Todas":
            df_search = df_search[df_search['archetype'].isin(POSITION_MAP[position])]
        
        if len(df_search) > 0:
            df_search['wk_score'] = df_search['performance_score'] + (max_age - df_search['Age'] + 1) * 2
            df_result = df_search.nlargest(20, 'wk_score')
            df_display = df_result[['Player', 'archetype', 'domestic_league', 'Age', 'performance_score', 'wk_score', 'market_value_current', 'gap_ratio']].copy()
            df_display.columns = ['Jugador', 'Arquetipo', 'Liga', 'Edad', 'Score', 'WK Score', 'MV (TM)', 'Gap']
            df_display['MV (TM)'] = df_display['MV (TM)'].apply(lambda x: f"€{x/1e6:.1f}M")
            df_display['Gap'] = df_display['Gap'].apply(lambda x: f"x{x:.2f}" if pd.notna(x) else "-")
            df_display[['Score', 'WK Score']] = df_display[['Score', 'WK Score']].round(1)
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("Sin resultados")

# MÓDULO 5
elif "Módulo 5" in module:
    st.title("🔄 Flip Opportunities")
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Presupuesto (M€)", 0.5, 50.0, 20.0, 2.0, key="f_budget") * 1_000_000
    with col2:
        max_age = st.slider("Edad máxima", 20, 28, 26, key="f_age")
    
    position = st.selectbox("Posición", ["Todas"] + list(POSITION_MAP.keys()), key="f_pos")
    season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True), key="f_season")
    
    if st.button("🔄 Buscar", type="primary"):
        df_search = df[
            (df['season'] == season) &
            (df['Age'] <= max_age) &
            (df['market_value_current'] <= budget) &
            (df['market_value_current'] >= 500_000) &
            (df['gap_ratio'] >= 1.3) &
            (df['performance_score'] >= 55) &
            (df['total_90s'] >= 15)
        ].copy()
        
        if position != "Todas":
            df_search = df_search[df_search['archetype'].isin(POSITION_MAP[position])]
        
        if len(df_search) > 0:
            df_search['flip_score'] = df_search['gap_ratio'] * 30 + df_search['performance_score'] * 0.5 + (max_age - df_search['Age']) * 2
            df_result = df_search.nlargest(20, 'flip_score')
            df_display = df_result[['Player', 'archetype', 'domestic_league', 'Age', 'performance_score', 'market_value_current', 'predicted_market_value', 'gap_ratio']].copy()
            df_display.columns = ['Jugador', 'Arquetipo', 'Liga', 'Edad', 'Score', 'MV (TM)', 'MV (Pred)', 'Gap']
            for col in ['MV (TM)', 'MV (Pred)']:
                df_display[col] = df_display[col].apply(lambda x: f"€{x/1e6:.1f}M")
            df_display['Gap'] = df_display['Gap'].apply(lambda x: f"x{x:.2f}")
            df_display['Score'] = df_display['Score'].round(1)
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("Sin resultados")
