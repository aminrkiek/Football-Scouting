import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances
import base64

st.set_page_config(
    page_title="Insight Scouting - AI Football Analytics",
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
    background: linear-gradient(135deg, #2D5F5D 0%, #48A999 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}
.module-card {
    border: 2px solid #48A999;
    border-radius: 15px;
    padding: 20px;
    background: linear-gradient(135deg, rgba(45, 95, 93, 0.1) 0%, rgba(72, 169, 153, 0.1) 100%);
    margin: 10px 0;
}
.score-badge {
    background: linear-gradient(135deg, #2D5F5D 0%, #48A999 100%);
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
.compare-badge {
    background: #ffc107;
    color: #000;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 0.8em;
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

# Session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_player_idx' not in st.session_state:
    st.session_state.selected_player_idx = None
if 'compare_players' not in st.session_state:
    st.session_state.compare_players = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

def create_radar_chart(player_row, archetype):
    weights = archetype_weights.get(archetype, {})
    top_stats = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:8]
    
    categories = []
    values = []
    
    for stat, weight in top_stats:
        pct_col = f'pct_{stat}'
        if pct_col in player_row.index:
            categories.append(stat.replace('_p90', '').replace('_', ' ').title()[:15])
            val = player_row[pct_col] if pd.notna(player_row[pct_col]) else 0
            values.append(val)
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Percentil',
        line=dict(color='#48A999', width=2),
        fillcolor='rgba(72, 169, 153, 0.3)'
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
        
        # Botones de acción
        col_a, col_b = st.columns(2)
        with col_a:
            player_id = f"{player_row['player_id']}_{player_row['season']}"
            if player_id in st.session_state.watchlist:
                if st.button("⭐ En watchlist", key=f"wl_{player_id}"):
                    st.session_state.watchlist.remove(player_id)
                    st.rerun()
            else:
                if st.button("☆ Agregar", key=f"add_{player_id}"):
                    st.session_state.watchlist.append(player_id)
                    st.rerun()
        
        with col_b:
            if player_id not in st.session_state.compare_players:
                if st.button("📊 Comparar", key=f"cmp_{player_id}"):
                    if len(st.session_state.compare_players) < 3:
                        st.session_state.compare_players.append(player_id)
                        st.rerun()
                    else:
                        st.warning("Máximo 3 jugadores")
        
        st.markdown("---")
        
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
        # FIX: Manejar NaN en score
        score = player_row['performance_score']
        if pd.notna(score):
            st.markdown(f"### 📊 Performance Score")
            st.progress(float(score)/100)
            st.markdown(f"<h1 style='text-align: center; color: #48A999;'>{score:.1f} / 100</h1>", 
                       unsafe_allow_html=True)
        else:
            st.warning("⚠️ Performance score no disponible para esta temporada")
        
        st.markdown("### 💰 Valoración")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if pd.notna(player_row['market_value_current']):
                st.metric("Transfermarkt", f"€{player_row['market_value_current']/1e6:.1f}M")
        with col_b:
            if pd.notna(player_row.get('predicted_market_value')):
                delta = player_row['predicted_market_value'] - player_row['market_value_current']
                st.metric("Predicho", f"€{player_row['predicted_market_value']/1e6:.1f}M",
                         delta=f"{delta/1e6:.1f}M")
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
        
        if pd.notna(score):
            st.markdown("### 📈 Estadísticas (Percentil vs arquetipo)")
            fig = create_radar_chart(player_row, player_row['archetype'])
            st.plotly_chart(fig, use_container_width=True)

def export_to_csv(df_results, filename="search_results.csv"):
    csv = df_results.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Descargar CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# Sidebar
try:
    st.sidebar.image("logo.png", width=200)
except:
    st.sidebar.title("⚽ Insight Scouting")

st.sidebar.markdown("### AI Football Analytics")
st.sidebar.markdown("---")

module = st.sidebar.radio(
    "📋 Navegación:",
    [
        "🏠 Inicio",
        "🔍 Búsqueda de Jugadores",
        "📊 Comparador",
        "⭐ Watchlist",
        "📜 Historial",
        "💰 Best by Budget",
        "🔄 Find Replacement",
        "💎 Undervalued",
        "⭐ Wonderkids",
        "🔄 Flip Opportunities",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"📊 **Dataset**\n"
    f"• Jugadores: {len(df):,}\n"
    f"• Temporadas: 2017-2025\n"
    f"• Ligas: 7 europeas"
)

# Watchlist counter
if len(st.session_state.watchlist) > 0:
    st.sidebar.success(f"⭐ Watchlist: {len(st.session_state.watchlist)} jugadores")

# MÓDULO HOME
if "Inicio" in module:
    st.markdown("<div class='big-title'>⚽ Insight Scouting</div>", unsafe_allow_html=True)
    st.markdown("### AI-Powered Football Analytics Platform")
    
    # Video intro
    try:
        st.video("intro.mp4")
    except:
        st.info("💡 Bienvenido al sistema de scouting más avanzado del mercado")
    
    st.markdown("---")
    
    # Stats globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.metric("🌍 Jugadores", f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.metric("⚽ Ligas", "7 Top")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        avg_score = df['performance_score'].mean()
        st.metric("📊 Score Medio", f"{avg_score:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        total_value = df['market_value_current'].sum() / 1e9
        st.metric("💰 Valor Total", f"€{total_value:.1f}B")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráficos de distribución
    st.markdown("## 📊 Análisis del Mercado")
    
    tab1, tab2, tab3 = st.tabs(["Distribución de Scores", "Precios por Liga", "Edad vs Valor"])
    
    with tab1:
        fig = px.histogram(
            df[df['performance_score'].notna()],
            x='performance_score',
            nbins=50,
            title='Distribución de Performance Scores',
            color_discrete_sequence=['#48A999']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        league_avg = df.groupby('domestic_league')['market_value_current'].mean().sort_values(ascending=False).head(7)
        fig = px.bar(
            x=league_avg.index,
            y=league_avg.values/1e6,
            title='Valor de Mercado Promedio por Liga (M€)',
            color_discrete_sequence=['#48A999']
        )
        fig.update_layout(xaxis_title="Liga", yaxis_title="Valor Promedio (M€)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        sample = df[df['market_value_current'].notna()].sample(min(1000, len(df)))
        fig = px.scatter(
            sample,
            x='Age',
            y='market_value_current',
            color='performance_score',
            size='market_value_current',
            title='Edad vs Valor de Mercado',
            color_continuous_scale='Teal'
        )
        fig.update_yaxis(type="log")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 🎯 Módulos Disponibles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='module-card'>
        <h3>🔍 Búsqueda de Jugadores</h3>
        <p>Busca cualquier jugador por nombre y visualiza su perfil completo.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>🔄 Find Replacement</h3>
        <p>Encuentra jugadores con perfil estadístico similar usando ML.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>⭐ Wonderkids</h3>
        <p>Identifica jóvenes talentos con alto potencial de revalorización.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='module-card'>
        <h3>💰 Best by Budget</h3>
        <p>Encuentra los mejores jugadores según performance dentro de un presupuesto.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>💎 Undervalued</h3>
        <p>Detecta jugadores cuyo rendimiento supera su precio de mercado.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>🔄 Flip Opportunities</h3>
        <p>Identifica jugadores infravalorados para estrategias de trading.</p>
        </div>
        """, unsafe_allow_html=True)

# MÓDULO COMPARADOR
elif "Comparador" in module:
    st.title("📊 Comparador de Jugadores")
    
    if len(st.session_state.compare_players) == 0:
        st.info("💡 Selecciona jugadores desde cualquier búsqueda para compararlos")
    else:
        st.success(f"✅ {len(st.session_state.compare_players)} jugadores seleccionados")
        
        if st.button("🗑️ Limpiar comparación"):
            st.session_state.compare_players = []
            st.rerun()
        
        # Obtener jugadores
        compare_data = []
        for pid in st.session_state.compare_players:
            player_id, season = pid.split('_')
            player = df[(df['player_id'] == int(player_id)) & (df['season'] == season)].iloc[0]
            compare_data.append(player)
        
        # Mostrar perfiles lado a lado
        cols = st.columns(len(compare_data))
        for idx, player in enumerate(compare_data):
            with cols[idx]:
                if pd.notna(player.get('player_image_url')):
                    st.image(player['player_image_url'], use_column_width=True)
                st.markdown(f"### {player['Player']}")
                st.metric("Score", f"{player['performance_score']:.1f}")
                st.metric("Valor", f"€{player['market_value_current']/1e6:.1f}M")
                st.metric("Edad", f"{int(player['Age'])}")
        
        # Radar comparativo
        st.markdown("### 📈 Comparación Estadística")
        fig = go.Figure()
        
        colors = ['#48A999', '#FF6B6B', '#4ECDC4']
        
        for idx, player in enumerate(compare_data):
            weights = archetype_weights.get(player['archetype'], {})
            top_stats = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:8]
            
            categories = []
            values = []
            
            for stat, _ in top_stats:
                pct_col = f'pct_{stat}'
                if pct_col in player.index:
                    categories.append(stat.replace('_p90', '').replace('_', ' ').title()[:15])
                    val = player[pct_col] if pd.notna(player[pct_col]) else 0
                    values.append(val)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=player['Player'][:15],
                line=dict(color=colors[idx], width=2)
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# MÓDULO WATCHLIST
elif "Watchlist" in module:
    st.title("⭐ Watchlist")
    
    if len(st.session_state.watchlist) == 0:
        st.info("💡 Tu watchlist está vacía. Agrega jugadores desde cualquier búsqueda.")
    else:
        st.success(f"✅ {len(st.session_state.watchlist)} jugadores guardados")
        
        if st.button("🗑️ Limpiar watchlist"):
            st.session_state.watchlist = []
            st.rerun()
        
        # Obtener jugadores de watchlist
        watchlist_players = []
        for pid in st.session_state.watchlist:
            player_id, season = pid.split('_')
            player = df[(df['player_id'] == int(player_id)) & (df['season'] == season)]
            if len(player) > 0:
                watchlist_players.append(player.iloc[0])
        
        # Mostrar en grid
        for i in range(0, len(watchlist_players), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(watchlist_players):
                    player = watchlist_players[i + j]
                    with col:
                        if pd.notna(player.get('player_image_url')):
                            st.image(player['player_image_url'], use_column_width=True)
                        st.markdown(f"**{player['Player'][:20]}**")
                        if pd.notna(player.get('performance_score')):
                            st.markdown(f"⭐ {player['performance_score']:.1f}")
                        st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                        
                        pid = f"{player['player_id']}_{player['season']}"
                        if st.button("🗑️", key=f"rm_{pid}"):
                            st.session_state.watchlist.remove(pid)
                            st.rerun()

# MÓDULO HISTORIAL
elif "Historial" in module:
    st.title("📜 Historial de Búsquedas")
    
    if len(st.session_state.search_history) == 0:
        st.info("💡 Tu historial está vacío. Realiza búsquedas para verlas aquí.")
    else:
        for idx, search in enumerate(reversed(st.session_state.search_history[-10:])):
            with st.expander(f"🔍 {search['type']} - {search['timestamp']}", expanded=(idx==0)):
                st.json(search['params'])

# MÓDULO BÚSQUEDA (igual que antes pero añade al historial)
elif "Búsqueda" in module:
    st.title("🔍 Búsqueda de Jugadores")
    
    search_name = st.text_input("🔎 Buscar jugador:", placeholder="Ej: Pedri, Goretzka...", key="player_search_main")
    
    col1, col2 = st.columns(2)
    with col1:
        search_season = st.selectbox("Temporada:", sorted(df['season'].unique(), reverse=True), key="search_season_main")
    with col2:
        search_liga = st.multiselect("Filtrar por liga:", df['domestic_league'].dropna().unique(), key="search_liga")
    
    if search_name:
        # Añadir al historial
        st.session_state.search_history.append({
            'type': 'Búsqueda directa',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'params': {'name': search_name, 'season': search_season}
        })
        
        df_search = df[df['Player'].str.contains(search_name, case=False, na=False)]
        
        if len(search_liga) > 0:
            df_search = df_search[df_search['domestic_league'].isin(search_liga)]
        
        if len(df_search) == 0:
            st.warning(f"No se encontraron jugadores")
        else:
            st.success(f"✅ {len(df_search)} resultados")
            
            for season in sorted(df_search['season'].unique(), reverse=True):
                season_data = df_search[df_search['season'] == season]
                if len(season_data) > 0:
                    st.markdown(f"### 📅 Temporada {season}")
                    for idx, player in season_data.iterrows():
                        with st.expander(f"⚽ {player['Player']}", expanded=(season==search_season)):
                            show_player_profile(player)

# MÓDULOS 1-5 (con exportar CSV y fixes)
elif "Best by Budget" in module:
    st.title("💰 Best by Budget")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        budget = st.number_input("Presupuesto (M€)", 0.5, 200.0, 30.0, 5.0) * 1_000_000
    with col2:
        position = st.selectbox("Posición", ["Todas"] + list(POSITION_MAP.keys()))
    with col3:
        season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True))
    with col4:
        sort_by = st.selectbox("Ordenar", ["Performance Score", "Market Value", "Gap Ratio"])
    
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
        
        if len(df_search) > 0:
            sort_map = {
                "Performance Score": "performance_score",
                "Market Value": "market_value_current",
                "Gap Ratio": "gap_ratio"
            }
            st.session_state.search_results = df_search.nlargest(20, sort_map[sort_by])
            st.session_state.selected_player_idx = None
        else:
            st.session_state.search_results = None
    
    if st.session_state.search_results is not None:
        df_result = st.session_state.search_results
        
        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp1:
            st.success(f"✅ {len(df_result)} jugadores")
        with col_exp2:
            export_to_csv(df_result[['Player', 'Age', 'archetype', 'performance_score', 
                                     'market_value_current', 'predicted_transfer_fee']])
        
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
                        st.markdown(f"<div class='score-badge'>{player['performance_score']:.1f}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='price-tag'>€{player['market_value_current']/1e6:.1f}M</div>", unsafe_allow_html=True)
                        if st.button("Ver", key=f"btn_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()

# MÓDULO 3 (sin mensaje info)
elif "Undervalued" in module:
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
            (df['total_90s'] >= 15) &
            (df['market_value_current'] >= 3_000_000)  # Filtro silencioso
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
        
        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp1:
            st.success(f"✅ {len(df_result)} oportunidades")
        with col_exp2:
            export_to_csv(df_result[['Player', 'Age', 'archetype', 'performance_score', 
                                     'market_value_current', 'gap_ratio']], "undervalued.csv")
        
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
                        st.success(f"💎 x{player['gap_ratio']:.2f}")
                        if pd.notna(player.get('performance_score')):
                            st.markdown(f"⭐ {player['performance_score']:.1f}")
                        st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                        if st.button("Ver", key=f"under_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()

# MÓDULO 5 (edad configurable)
elif "Flip Opportunities" in module:
    st.title("🔄 Flip Opportunities")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input("Presupuesto (M€)", 0.5, 50.0, 20.0, 2.0) * 1_000_000
    with col2:
        min_age = st.slider("Edad mínima", 16, 35, 18)
    with col3:
        max_age = st.slider("Edad máxima", 16, 35, 26)
    
    position = st.selectbox("Posición", ["Todas"] + list(POSITION_MAP.keys()), key="f_pos")
    season = st.selectbox("Temporada", sorted(df['season'].unique(), reverse=True), key="f_season")
    
    if st.button("🔄 Buscar", type="primary"):
        df_search = df[
            (df['season'] == season) &
            (df['Age'] >= min_age) &
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
            df_search['flip_score'] = (df_search['gap_ratio'] * 30 + 
                                       df_search['performance_score'] * 0.5 + 
                                       (max_age - df_search['Age']) * 2)
            st.session_state.search_results = df_search.nlargest(16, 'flip_score')
            st.session_state.selected_player_idx = None
        else:
            st.session_state.search_results = None
    
    if st.session_state.search_results is not None:
        df_result = st.session_state.search_results
        
        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp1:
            st.success(f"✅ {len(df_result)} oportunidades")
        with col_exp2:
            export_to_csv(df_result[['Player', 'Age', 'performance_score', 
                                     'market_value_current', 'gap_ratio']], "flip_opportunities.csv")
        
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
                        if pd.notna(player.get('performance_score')):
                            st.markdown(f"⭐ {player['performance_score']:.1f}")
                        st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                        if st.button("Ver", key=f"flip_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()

# MÓDULOS 2 y 4 (implementación similar pero compacta)
