import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(
    page_title="Primeira Analytics - Scouting System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-title {
    font-size: 3em;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}
.module-card {
    border: 2px solid #667eea;
    border-radius: 15px;
    padding: 20px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    margin: 10px 0;
}
.score-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}
.price-tag {
    color: #2ecc71;
    font-size: 1.2em;
    font-weight: bold;
}
.stat-box {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    file_id = "1PY4vezJf599CGLJxfS6KRvCSjiJGb3Va"
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

# Session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_player_idx' not in st.session_state:
    st.session_state.selected_player_idx = None
if 'compare_players' not in st.session_state:
    st.session_state.compare_players = []

def create_radar_chart(player_row, archetype):
    weights = archetype_weights.get(archetype, {})
    top_stats = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:8]
    
    categories = []
    values = []
    
    for stat, weight in top_stats:
        pct_col = f'pct_{stat}'
        if pct_col in player_row.index:
            categories.append(stat.replace('_p90', '').replace('_', ' ').title()[:15])
            values.append(player_row[pct_col] if pd.notna(player_row[pct_col]) else 0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Percentil',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def show_player_profile(player_row):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if pd.notna(player_row.get('player_image_url')):
            st.image(player_row['player_image_url'], width=150)
        else:
            st.markdown("# 🧑‍🦱")
        
        st.markdown(f"### {player_row['Player']}")
        st.markdown(f"**{player_row['archetype']}**")
        
        if pd.notna(player_row.get('citizenship')):
            st.markdown(f"🌍 {player_row['citizenship']}")
        if pd.notna(player_row.get('current_club_name')):
            st.markdown(f"⚽ {player_row['current_club_name']}")
        if pd.notna(player_row.get('domestic_league')):
            st.markdown(f"🏆 {player_row['domestic_league']}")
        if pd.notna(player_row.get('height')):
            st.markdown(f"📏 {player_row['height']} cm")
        if pd.notna(player_row.get('foot')):
            st.markdown(f"🦶 {player_row['foot']}")
        
        st.markdown(f"🎂 {int(player_row['Age'])} años")
        st.markdown(f"⚽ {player_row['total_90s']:.0f} partidos (90')")
    
    with col2:
        score = player_row['performance_score']
        st.markdown(f"### 📊 Performance Score")
        st.progress(score/100)
        st.markdown(f"<h1 style='text-align: center; color: #667eea;'>{score:.1f} / 100</h1>", 
                   unsafe_allow_html=True)
        
        st.markdown("### 💰 Valoración")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Transfermarkt", f"€{player_row['market_value_current']/1e6:.1f}M")
        with col_b:
            st.metric("Predicho", f"€{player_row['predicted_market_value']/1e6:.1f}M",
                     delta=f"{(player_row['predicted_market_value']-player_row['market_value_current'])/1e6:.1f}M")
        with col_c:
            if pd.notna(player_row.get('predicted_transfer_fee')):
                st.metric("Transfer Fee", f"€{player_row['predicted_transfer_fee']/1e6:.1f}M")
        
        if pd.notna(player_row.get('gap_ratio')):
            gap = player_row['gap_ratio']
            if gap > 1.5:
                st.success(f"💎 Muy infravalorado: x{gap:.2f}")
            elif gap > 1.3:
                st.info(f"💰 Infravalorado: x{gap:.2f}")
            elif gap < 0.7:
                st.error(f"⚠️ Sobrevalorado: x{gap:.2f}")
        
        st.markdown("### 📈 Estadísticas (Percentil vs arquetipo)")
        fig = create_radar_chart(player_row, player_row['archetype'])
        st.plotly_chart(fig, use_container_width=True)

# Sidebar
st.sidebar.title("⚽ Primeira Analytics")
st.sidebar.markdown("---")

module = st.sidebar.radio(
    "📋 Navegación:",
    [
        "🏠 Inicio",
        "🔍 Búsqueda de Jugadores",
        "💰 Módulo 1: Best by Budget",
        "🔄 Módulo 2: Find Replacement",
        "💎 Módulo 3: Undervalued",
        "⭐ Módulo 4: Wonderkids",
        "🔄 Módulo 5: Flip Opportunities",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"📊 **Dataset**\n"
    f"• Jugadores: {len(df):,}\n"
    f"• Temporadas: 2017-2025\n"
    f"• Ligas: 7 europeas\n"
    f"• Últimos datos: 2024-2025"
)

# MÓDULO HOME
if "Inicio" in module:
    st.markdown("<div class='big-title'>⚽ Primeira Analytics</div>", unsafe_allow_html=True)
    st.markdown("### Sistema de Scouting Inteligente con Machine Learning")
    
    st.markdown("---")
    
    # Stats globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.metric("🌍 Jugadores", f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.metric("⚽ Ligas", "7 Top Europeas")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        avg_score = df['performance_score'].mean()
        st.metric("📊 Score Promedio", f"{avg_score:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        total_value = df['market_value_current'].sum() / 1e9
        st.metric("💰 Valor Total", f"€{total_value:.1f}B")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Módulos
    st.markdown("## 🎯 Módulos Disponibles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='module-card'>
        <h3>🔍 Búsqueda de Jugadores</h3>
        <p><strong>¿Para qué sirve?</strong></p>
        <p>Busca cualquier jugador por nombre y visualiza su perfil completo con estadísticas, valoración de mercado y radar de rendimiento.</p>
        <p><strong>Casos de uso:</strong></p>
        <ul>
        <li>Evaluar un jugador antes de una negociación</li>
        <li>Verificar el rendimiento actual vs valoración</li>
        <li>Comparar estadísticas entre temporadas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>🔄 Módulo 2: Find Replacement</h3>
        <p><strong>¿Para qué sirve?</strong></p>
        <p>Encuentra jugadores con perfil estadístico similar a uno de referencia. Utiliza similitud euclidiana en espacio de percentiles ponderados por importancia del arquetipo.</p>
        <p><strong>Casos de uso:</strong></p>
        <ul>
        <li>Reemplazar una salida confirmada</li>
        <li>Buscar alternativas más económicas</li>
        <li>Identificar perfiles tácticos equivalentes</li>
        </ul>
        <p><strong>Ejemplo:</strong> "Necesito reemplazar a Goretzka con presupuesto de €15M"</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>⭐ Módulo 4: Wonderkids</h3>
        <p><strong>¿Para qué sirve?</strong></p>
        <p>Identifica jóvenes talentos con alto rendimiento relativo a su edad. Combina performance score con bonus por juventud.</p>
        <p><strong>Casos de uso:</strong></p>
        <ul>
        <li>Inversión en promesas sub-21</li>
        <li>Refuerzo de cantera/filial</li>
        <li>Apuestas de futuro con reventa</li>
        </ul>
        <p><strong>Filtros:</strong> Edad máxima, presupuesto, posición</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='module-card'>
        <h3>💰 Módulo 1: Best by Budget</h3>
        <p><strong>¿Para qué sirve?</strong></p>
        <p>Encuentra los mejores jugadores según performance score dentro de un presupuesto específico.</p>
        <p><strong>Casos de uso:</strong></p>
        <ul>
        <li>Refuerzo de plantilla con presupuesto limitado</li>
        <li>Maximizar calidad por euro invertido</li>
        <li>Análisis de mercado por posición</li>
        </ul>
        <p><strong>Ejemplo:</strong> "Mejores centrocampistas por €30M"</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>💎 Módulo 3: Undervalued Players</h3>
        <p><strong>¿Para qué sirve?</strong></p>
        <p>Detecta jugadores cuyo rendimiento supera significativamente su precio de mercado (gap ratio > 1.5). Oportunidades de mercado.</p>
        <p><strong>Casos de uso:</strong></p>
        <ul>
        <li>Gangas de mercado (alto ROI)</li>
        <li>Fichajes bajo el radar</li>
        <li>Inversiones con potencial de revalorización</li>
        </ul>
        <p><strong>Métrica clave:</strong> Gap Ratio = Valor Predicho / Valor Actual</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>🔄 Módulo 5: Flip Opportunities</h3>
        <p><strong>¿Para qué sirve?</strong></p>
        <p>Identifica jugadores infravalorados en edad óptima (20-26) para comprar-revender. Combina gap, rendimiento y potencial de revalorización.</p>
        <p><strong>Casos de uso:</strong></p>
        <ul>
        <li>Estrategia de trading de jugadores</li>
        <li>Compras con plusvalía garantizada</li>
        <li>Gestión patrimonial del club</li>
        </ul>
        <p><strong>Filtros:</strong> Presupuesto, edad máxima, gap mínimo 1.3x</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Metodología")
    
    with st.expander("🧠 Performance Score"):
        st.markdown("""
        **Cálculo del Performance Score (0-100):**
        
        1. **Ajuste por calidad de liga**: Stats p90 × quality_factor
        2. **Percentiles por arquetipo**: Ranking dentro del arquetipo (0-100)
        3. **Pesos específicos**: Cada arquetipo pondera stats diferentes
           - Ej: CF → Goles, xG, Remates
           - Ej: CDM → Tkl+Int, Recuperaciones, Intercepciones
        4. **Normalización final**: Score 0-100 comparable entre arquetipos
        
        **Validación**: Correlación con market_value ~0.5-0.8 según posición
        """)
    
    with st.expander("💰 Predicción de Valores"):
        st.markdown("""
        **Modelo de Transfer Fee:**
        - Algoritmo: Random Forest (69 features)
        - Entrenado en: 2,829 transferencias reales
        - MAE: ~€3.2M
        
        **Modelo de Market Value:**
        - Algoritmo: XGBoost optimizado (Optuna)
        - Features: Stats p90, edad, liga, competición europea
        - R²: 0.87
        
        **Gap Ratio** = Predicted MV / Current MV
        - > 1.5 → Muy infravalorado 💎
        - 1.3-1.5 → Infravalorado 💰
        - 0.7-1.3 → Valoración justa ✅
        - < 0.7 → Sobrevalorado ⚠️
        """)
    
    with st.expander("🎯 Arquetipos (24 perfiles)"):
        st.markdown("""
        **Porteros (2):** Ball-Playing, Traditional
        
        **Defensas centrales (3):** Ball-Playing, Complete, Traditional
        
        **Laterales (3):** Balanced, Complete, Defensive
        
        **Pivotes (2):** Ball-Winner, Deep-Lying Playmaker
        
        **Centrocampistas (3):** Advanced Playmaker, Box-to-Box, Defensive-Minded
        
        **Mediapuntas (4):** Balanced, Complete, Shadow Striker, Support
        
        **Delanteros (4):** Balanced, Clinical Finisher, False 9, Support
        
        **Extremos (3):** Balanced, Complete, Support
        """)

# MÓDULO BÚSQUEDA
elif "Búsqueda" in module:
    st.title("🔍 Búsqueda de Jugadores")
    
    search_name = st.text_input(
        "🔎 Buscar jugador por nombre:",
        placeholder="Ej: Pedri, Goretzka, Mbappé...",
        key="player_search_main"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        search_season = st.selectbox(
            "Temporada:",
            sorted(df['season'].unique(), reverse=True),
            key="search_season_main"
        )
    with col2:
        search_liga = st.multiselect(
            "Filtrar por liga (opcional):",
            df['domestic_league'].dropna().unique(),
            key="search_liga"
        )
    
    if search_name:
        df_search = df[df['Player'].str.contains(search_name, case=False, na=False)]
        
        if len(search_liga) > 0:
            df_search = df_search[df_search['domestic_league'].isin(search_liga)]
        
        if len(df_search) == 0:
            st.warning(f"No se encontraron jugadores con '{search_name}'")
        else:
            st.success(f"✅ {len(df_search)} resultados encontrados")
            
            # Agrupar por temporada
            seasons_available = df_search['season'].unique()
            
            for season in sorted(seasons_available, reverse=True):
                season_data = df_search[df_search['season'] == season]
                
                if len(season_data) > 0:
                    st.markdown(f"### 📅 Temporada {season}")
                    
                    for idx, player in season_data.iterrows():
                        with st.expander(f"⚽ {player['Player']} - {player['archetype']}", expanded=(season==search_season)):
                            show_player_profile(player)

# MÓDULO 1
elif "Módulo 1" in module:
    st.title("💰 Best by Budget")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        budget = st.number_input("Presupuesto (M€)", 0.5, 200.0, 30.0, 5.0) * 1_000_000
    with col2:
        position = st.selectbox("Posición", ["Todas"] + list(POSITION_MAP.keys()))
    with col3:
        season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True))
    with col4:
        sort_by = st.selectbox("Ordenar por", ["Performance Score", "Market Value", "Transfer Fee", "Gap Ratio"])
    
    min_90s = st.slider("Mínimo 90 minutos", 5, 40, 15)
    
    # Filtros avanzados (colapsable)
    with st.expander("⚙️ Filtros avanzados"):
        col_a, col_b = st.columns(2)
        with col_a:
            selected_leagues = st.multiselect("Ligas:", df['domestic_league'].dropna().unique())
        with col_b:
            age_range = st.slider("Edad:", 16, 40, (18, 35))
    
    if st.button("🔍 Buscar", type="primary", key="search_m1"):
        df_search = df[
            (df['season'] == season) &
            (df['market_value_current'] <= budget) &
            (df['performance_score'] >= 60) &
            (df['total_90s'] >= min_90s) &
            (df['Age'] >= age_range[0]) &
            (df['Age'] <= age_range[1]) &
            (df['market_value_current'].notna())
        ].copy()
        
        if position != "Todas":
            df_search = df_search[df_search['archetype'].isin(POSITION_MAP[position])]
        
        if len(selected_leagues) > 0:
            df_search = df_search[df_search['domestic_league'].isin(selected_leagues)]
        
        if len(df_search) > 0:
            sort_map = {
                "Performance Score": "performance_score",
                "Market Value": "market_value_current",
                "Transfer Fee": "predicted_transfer_fee",
                "Gap Ratio": "gap_ratio"
            }
            st.session_state.search_results = df_search.nlargest(20, sort_map[sort_by])
            st.session_state.selected_player_idx = None
        else:
            st.session_state.search_results = None
            st.warning("Sin resultados")
    
    # Mostrar resultados
    if st.session_state.search_results is not None:
        df_result = st.session_state.search_results
        
        if st.session_state.selected_player_idx is not None:
            st.markdown("---")
            player = df_result.iloc[st.session_state.selected_player_idx]
            show_player_profile(player)
            
            if st.button("⬅️ Volver a resultados"):
                st.session_state.selected_player_idx = None
                st.rerun()
            st.markdown("---")
        
        st.success(f"✅ {len(df_result)} jugadores encontrados")
        
        # Grid
        for i in range(0, len(df_result), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(df_result):
                    player = df_result.iloc[i + j]
                    with col:
                        if pd.notna(player.get('player_image_url')):
                            st.image(player['player_image_url'], use_column_width=True)
                        
                        st.markdown(f"**{player['Player'][:20]}**")
                        st.markdown(f"<div class='score-badge'>{player['performance_score']:.1f}</div>", 
                                   unsafe_allow_html=True)
                        st.markdown(f"<div class='price-tag'>€{player['market_value_current']/1e6:.1f}M</div>", 
                                   unsafe_allow_html=True)
                        st.caption(f"{player['archetype'][:25]}...")
                        
                        if st.button(f"Ver perfil", key=f"btn_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()

# MÓDULO 2
elif "Módulo 2" in module:
    st.title("🔄 Find Replacement")
    
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Nombre del jugador", placeholder="Ej: Pedri")
    with col2:
        budget = st.number_input("Presupuesto (M€) - opcional", 0.0, 200.0, 0.0, 5.0)
    
    season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True), key="r_season")
    
    if st.button("🔍 Buscar", type="primary", key="search_repl"):
        if not player_name:
            st.warning("Ingresa un nombre")
        else:
            ref_candidates = df[df['Player'].str.contains(player_name, case=False, na=False)]
            if len(ref_candidates) == 0:
                st.error(f"Jugador '{player_name}' no encontrado")
                st.session_state.search_results = None
            else:
                ref = ref_candidates[ref_candidates['season'] == season].iloc[0] if len(ref_candidates[ref_candidates['season'] == season]) > 0 else ref_candidates.iloc[0]
                
                arch_weights = archetype_weights.get(ref['archetype'], {})
                top_features = [f'pct_{f}' for f, _ in sorted(arch_weights.items(), key=lambda x: x[1], reverse=True)[:15] if f'pct_{f}' in df.columns]
                
                if top_features:
                    ref_vector = ref[top_features].fillna(50).values.reshape(1, -1)
                    
                    df_pool = df[
                        (df['season'] == season) &
                        (df['archetype'] == ref['archetype']) &
                        (~df['Player'].str.contains(player_name, case=False, na=False)) &
                        (df['total_90s'] >= 10)
                    ].copy()
                    
                    if budget > 0:
                        df_pool = df_pool[df_pool['market_value_current'] <= budget * 1e6]
                    
                    if len(df_pool) > 0:
                        pool_vectors = df_pool[top_features].fillna(50).values
                        
                        feature_weights = np.array([arch_weights.get(f.replace('pct_', ''), 1/len(top_features)) for f in top_features])
                        feature_weights = feature_weights / feature_weights.sum()
                        
                        diff = pool_vectors - ref_vector
                        distances = np.sqrt(np.sum((diff ** 2) * feature_weights, axis=1))
                        similarity = ((1 - distances / distances.max()) * 100).round(1)
                        
                        df_pool['similarity'] = similarity
                        
                        st.session_state.search_results = df_pool.nlargest(12, 'similarity')
                        st.session_state.reference_player = ref
                        st.session_state.selected_player_idx = None
                    else:
                        st.session_state.search_results = None
    
    if st.session_state.search_results is not None and 'reference_player' in st.session_state:
        ref = st.session_state.reference_player
        
        st.markdown("### 📌 Jugador de referencia:")
        with st.container():
            show_player_profile(ref)
        
        st.markdown("---")
        
        if st.session_state.selected_player_idx is not None:
            player = st.session_state.search_results.iloc[st.session_state.selected_player_idx]
            st.markdown("### 🔄 Perfil del jugador seleccionado:")
            show_player_profile(player)
            
            if st.button("⬅️ Volver a alternativas"):
                st.session_state.selected_player_idx = None
                st.rerun()
            
            st.markdown("---")
        
        st.markdown("### 🔄 Alternativas similares:")
        df_result = st.session_state.search_results
        
        for i in range(0, len(df_result), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(df_result):
                    player = df_result.iloc[i + j]
                    with col:
                        if pd.notna(player.get('player_image_url')):
                            st.image(player['player_image_url'], use_column_width=True)
                        
                        st.markdown(f"**{player['Player'][:20]}**")
                        st.markdown(f"🎯 **{player['similarity']:.1f}%**")
                        st.markdown(f"⭐ {player['performance_score']:.1f}")
                        st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                        
                        if st.button(f"Ver perfil", key=f"repl_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()

# MÓDULO 3 (con filtro €3M mínimo)
elif "Módulo 3" in module:
    st.title("💎 Undervalued Players")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        position = st.selectbox("Posición", ["Todas"] + list(POSITION_MAP.keys()), key="u_pos")
    with col2:
        min_gap = st.slider("Gap mínimo", 1.0, 5.0, 1.5, 0.1)
    with col3:
        season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True), key="u_season")
    
    st.info("ℹ️ Se filtran jugadores con valor < €3M para evitar datos inconsistentes")
    
    if st.button("💎 Buscar", type="primary", key="search_under"):
        df_search = df[
            (df['season'] == season) &
            (df['gap_ratio'] >= min_gap) &
            (df['performance_score'] >= 60) &
            (df['total_90s'] >= 15) &
            (df['market_value_current'] >= 3_000_000)  # ← FIX: Filtro €3M mínimo
        ].copy()
        
        if position != "Todas":
            df_search = df_search[df_search['archetype'].isin(POSITION_MAP[position])]
        
        if len(df_search) > 0:
            st.session_state.search_results = df_search.nlargest(16, 'gap_ratio')
            st.session_state.selected_player_idx = None
        else:
            st.session_state.search_results = None
    
    if st.session_state.search_results is not None:
        df_result = st.session_state.search_results
        
        if st.session_state.selected_player_idx is not None:
            player = df_result.iloc[st.session_state.selected_player_idx]
            show_player_profile(player)
            if st.button("⬅️ Volver"):
                st.session_state.selected_player_idx = None
                st.rerun()
            st.markdown("---")
        
        st.success(f"✅ {len(df_result)} oportunidades encontradas")
        
        for i in range(0, len(df_result), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(df_result):
                    player = df_result.iloc[i + j]
                    with col:
                        if pd.notna(player.get('player_image_url')):
                            st.image(player['player_image_url'], use_column_width=True)
                        st.markdown(f"**{player['Player'][:20]}**")
                        st.success(f"💎 x{player['gap_ratio']:.2f}")
                        st.markdown(f"⭐ {player['performance_score']:.1f}")
                        st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                        if st.button(f"Ver perfil", key=f"under_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()

# MÓDULO 4 y 5 (igual estructura anterior pero más compacto)
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
    
    if st.button("⭐ Buscar", type="primary", key="search_wk"):
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
            st.session_state.search_results = df_search.nlargest(16, 'wk_score')
            st.session_state.selected_player_idx = None
        else:
            st.session_state.search_results = None
    
    if st.session_state.search_results is not None:
        df_result = st.session_state.search_results
        
        if st.session_state.selected_player_idx is not None:
            player = df_result.iloc[st.session_state.selected_player_idx]
            show_player_profile(player)
            if st.button("⬅️ Volver"):
                st.session_state.selected_player_idx = None
                st.rerun()
            st.markdown("---")
        
        for i in range(0, len(df_result), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(df_result):
                    player = df_result.iloc[i + j]
                    with col:
                        if pd.notna(player.get('player_image_url')):
                            st.image(player['player_image_url'], use_column_width=True)
                        st.markdown(f"**{player['Player'][:20]}**")
                        st.info(f"🎂 {int(player['Age'])} años")
                        st.markdown(f"⭐ {player['wk_score']:.1f}")
                        st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                        if st.button(f"Ver perfil", key=f"wk_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()

elif "Módulo 5" in module:
    st.title("🔄 Flip Opportunities")
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Presupuesto (M€)", 0.5, 50.0, 20.0, 2.0, key="f_budget") * 1_000_000
    with col2:
        max_age = st.slider("Edad máxima", 20, 28, 26, key="f_age")
    
    position = st.selectbox("Posición", ["Todas"] + list(POSITION_MAP.keys()), key="f_pos")
    season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True), key="f_season")
    
    if st.button("🔄 Buscar", type="primary", key="search_flip"):
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
            st.session_state.search_results = df_search.nlargest(16, 'flip_score')
            st.session_state.selected_player_idx = None
        else:
            st.session_state.search_results = None
    
    if st.session_state.search_results is not None:
        df_result = st.session_state.search_results
        
        if st.session_state.selected_player_idx is not None:
            player = df_result.iloc[st.session_state.selected_player_idx]
            show_player_profile(player)
            if st.button("⬅️ Volver"):
                st.session_state.selected_player_idx = None
                st.rerun()
            st.markdown("---")
        
        for i in range(0, len(df_result), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(df_result):
                    player = df_result.iloc[i + j]
                    with col:
                        if pd.notna(player.get('player_image_url')):
                            st.image(player['player_image_url'], use_column_width=True)
                        st.markdown(f"**{player['Player'][:20]}**")
                        st.success(f"🔄 x{player['gap_ratio']:.2f}")
                        st.markdown(f"⭐ {player['performance_score']:.1f}")
                        st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                        if st.button(f"Ver perfil", key=f"flip_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()
