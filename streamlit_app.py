import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(
    page_title="Primeira Analytics - Scouting System",
    page_icon="⚽",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
.player-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.player-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.player-img {
    border-radius: 50%;
    width: 100px;
    height: 100px;
    object-fit: cover;
    margin: 10px auto;
}
.score-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
    margin: 5px 0;
}
.price-tag {
    color: #2ecc71;
    font-size: 1.2em;
    font-weight: bold;
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

# Función para crear radar chart
def create_radar_chart(player_row, archetype):
    weights = archetype_weights.get(archetype, {})
    top_stats = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:8]
    
    categories = []
    values = []
    
    for stat, weight in top_stats:
        pct_col = f'pct_{stat}'
        if pct_col in player_row.index:
            categories.append(stat.replace('_p90', '').replace('_', ' ').title())
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
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# Función para mostrar perfil de jugador
def show_player_profile(player_row):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Foto
        if pd.notna(player_row.get('player_image_url')):
            st.image(player_row['player_image_url'], width=150)
        else:
            st.markdown("🧑‍🦱")
        
        # Info básica
        st.markdown(f"### {player_row['Player']}")
        st.markdown(f"**{player_row['archetype']}**")
        
        if pd.notna(player_row.get('citizenship')):
            st.markdown(f"🌍 {player_row['citizenship']}")
        if pd.notna(player_row.get('current_club_name')):
            st.markdown(f"⚽ {player_row['current_club_name']}")
        if pd.notna(player_row.get('height')):
            st.markdown(f"📏 {player_row['height']} cm")
        if pd.notna(player_row.get('foot')):
            st.markdown(f"🦶 {player_row['foot']}")
        
        st.markdown(f"🎂 {int(player_row['Age'])} años")
        st.markdown(f"⚽ {player_row['total_90s']:.0f} partidos (90')")
    
    with col2:
        # Performance Score
        score = player_row['performance_score']
        st.markdown(f"### 📊 Performance Score")
        st.progress(score/100)
        st.markdown(f"<h1 style='text-align: center; color: #667eea;'>{score:.1f} / 100</h1>", 
                   unsafe_allow_html=True)
        
        # Valoración
        st.markdown("### 💰 Valoración")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Transfermarkt", f"€{player_row['market_value_current']/1e6:.1f}M")
        with col_b:
            st.metric("Predicho", f"€{player_row['predicted_market_value']/1e6:.1f}M")
        with col_c:
            if pd.notna(player_row.get('predicted_transfer_fee')):
                st.metric("Transfer Fee", f"€{player_row['predicted_transfer_fee']/1e6:.1f}M")
        
        if pd.notna(player_row.get('gap_ratio')):
            gap = player_row['gap_ratio']
            if gap > 1.3:
                st.success(f"💎 Infravalorado: x{gap:.2f}")
            elif gap < 0.7:
                st.warning(f"⚠️ Sobrevalorado: x{gap:.2f}")
        
        # Radar chart
        st.markdown("### 📈 Estadísticas (Percentil)")
        fig = create_radar_chart(player_row, player_row['archetype'])
        st.plotly_chart(fig, use_container_width=True)

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

# MÓDULO 1: BEST BY BUDGET
if "Módulo 1" in module:
    st.title("🔍 Best by Budget")
    st.markdown("Encuentra los mejores jugadores dentro de un presupuesto")
    
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
            
            st.success(f"✅ {len(df_result)} jugadores encontrados")
            
            # Mostrar en cards
            for i in range(0, len(df_result), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(df_result):
                        player = df_result.iloc[i + j]
                        with col:
                            # Card
                            if pd.notna(player.get('player_image_url')):
                                st.image(player['player_image_url'], use_column_width=True)
                            
                            st.markdown(f"**{player['Player'][:20]}**")
                            st.markdown(f"<div class='score-badge'>{player['performance_score']:.1f}</div>", 
                                       unsafe_allow_html=True)
                            st.markdown(f"<div class='price-tag'>€{player['market_value_current']/1e6:.1f}M</div>", 
                                       unsafe_allow_html=True)
                            st.caption(f"{player['archetype'][:25]}...")
                            
                            if st.button(f"Ver perfil", key=f"profile_{i}_{j}"):
                                st.session_state[f'show_profile_{i}_{j}'] = True
                            
                            # Mostrar perfil si se clickeó
                            if st.session_state.get(f'show_profile_{i}_{j}', False):
                                with st.expander("📋 Perfil completo", expanded=True):
                                    show_player_profile(player)
                                    if st.button("Cerrar", key=f"close_{i}_{j}"):
                                        st.session_state[f'show_profile_{i}_{j}'] = False
                                        st.rerun()

# MÓDULO 2: FIND REPLACEMENT
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
                
                # Mostrar jugador de referencia
                st.markdown("### 📌 Jugador de referencia:")
                with st.container():
                    show_player_profile(ref)
                
                st.markdown("---")
                st.markdown("### 🔄 Alternativas similares:")
                
                # Calcular similitud
                pct_cols = [f'pct_{f}' for f in ALL_PERF_FEATURES if f'pct_{f}' in df.columns]
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
                        df_result = df_pool.nlargest(12, 'similarity')
                        
                        # Mostrar en grid
                        for i in range(0, len(df_result), 4):
                            cols = st.columns(4)
                            for j, col in enumerate(cols):
                                if i + j < len(df_result):
                                    player = df_result.iloc[i + j]
                                    with col:
                                        if pd.notna(player.get('player_image_url')):
                                            st.image(player['player_image_url'], use_column_width=True)
                                        
                                        st.markdown(f"**{player['Player'][:20]}**")
                                        st.markdown(f"🎯 Similitud: **{player['similarity']:.1f}%**")
                                        st.markdown(f"⭐ Score: {player['performance_score']:.1f}")
                                        st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                                        
                                        if st.button(f"Ver perfil", key=f"repl_{i}_{j}"):
                                            st.session_state[f'show_repl_{i}_{j}'] = True
                                        
                                        if st.session_state.get(f'show_repl_{i}_{j}', False):
                                            with st.expander("📋 Perfil", expanded=True):
                                                show_player_profile(player)
                                                if st.button("Cerrar", key=f"close_repl_{i}_{j}"):
                                                    st.session_state[f'show_repl_{i}_{j}'] = False
                                                    st.rerun()
                    else:
                        st.warning("Sin alternativas con estos filtros")

# MÓDULO 3: UNDERVALUED
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
            df_result = df_search.nlargest(16, 'gap_ratio')
            
            for i in range(0, len(df_result), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(df_result):
                        player = df_result.iloc[i + j]
                        with col:
                            if pd.notna(player.get('player_image_url')):
                                st.image(player['player_image_url'], use_column_width=True)
                            
                            st.markdown(f"**{player['Player'][:20]}**")
                            st.success(f"💎 Gap: x{player['gap_ratio']:.2f}")
                            st.markdown(f"⭐ {player['performance_score']:.1f}")
                            st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                            
                            if st.button(f"Ver perfil", key=f"under_{i}_{j}"):
                                with st.expander("📋 Perfil", expanded=True):
                                    show_player_profile(player)
        else:
            st.warning("Sin resultados")

# MÓDULO 4: WONDERKIDS
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
            df_result = df_search.nlargest(16, 'wk_score')
            
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
                            st.markdown(f"⭐ WK: {player['wk_score']:.1f}")
                            st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                            
                            if st.button(f"Ver perfil", key=f"wk_{i}_{j}"):
                                with st.expander("📋 Perfil", expanded=True):
                                    show_player_profile(player)
        else:
            st.warning("Sin resultados")

# MÓDULO 5: FLIP OPPORTUNITIES
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
            df_result = df_search.nlargest(16, 'flip_score')
            
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
                                with st.expander("📋 Perfil", expanded=True):
                                    show_player_profile(player)
        else:
            st.warning("Sin resultados")
