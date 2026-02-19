import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances
import base64
import io

st.set_page_config(
    page_title="Insight Scouting - AI Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS + Intro
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
if 'show_intro' not in st.session_state:
    st.session_state.show_intro = True

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
                        st.warning("Máximo 3")
        
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
        score = player_row.get('performance_score')
        if pd.notna(score) and score > 0:
            st.markdown(f"### 📊 Performance Score")
            st.progress(float(score)/100)
            st.markdown(f"<h1 style='text-align: center; color: #48A999;'>{score:.1f} / 100</h1>", 
                       unsafe_allow_html=True)
        else:
            st.warning("⚠️ Performance score no disponible")
        
        st.markdown("### 💰 Valoración")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if pd.notna(player_row.get('market_value_current')):
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
        
        if pd.notna(score) and score > 0:
            st.markdown("### 📈 Estadísticas (Percentil vs arquetipo)")
            fig = create_radar_chart(player_row, player_row['archetype'])
            st.plotly_chart(fig, use_container_width=True)

def export_to_csv(df_results, filename="search_results.csv"):
    try:
        csv = df_results.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Descargar CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al exportar: {str(e)}")

def add_to_history(search_type, params):
    """Añadir búsqueda al historial"""
    st.session_state.search_history.append({
        'type': search_type,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'params': params
    })
    # Mantener solo últimos 50
    if len(st.session_state.search_history) > 50:
        st.session_state.search_history = st.session_state.search_history[-50:]

# Sidebar
try:
    st.sidebar.image("logo.png", width=200)
except:
    st.sidebar.markdown("# ⚽ Insight Scouting")

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

if len(st.session_state.watchlist) > 0:
    st.sidebar.success(f"⭐ Watchlist: {len(st.session_state.watchlist)}")

# Limpiar resultados al cambiar de módulo
if 'last_module' not in st.session_state:
    st.session_state.last_module = module

if st.session_state.last_module != module:
    st.session_state.search_results = None
    st.session_state.selected_player_idx = None
    st.session_state.last_module = module

# MÓDULO HOME con intro
if "Inicio" in module:
    # Intro animada (solo primera vez)
    if st.session_state.show_intro:
        try:
            video_file = open('intro.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes, autoplay=True)
            video_file.close()
        except:
            pass
        
        # Botón para saltar intro
        if st.button("⏭️ Saltar intro"):
            st.session_state.show_intro = False
            st.rerun()
        
        st.markdown("---")
    
    st.markdown("<div class='big-title'>⚽ Insight Scouting</div>", unsafe_allow_html=True)
    st.markdown("### Sistema de Scouting Inteligente: Valoración de Activos Deportivos mediante Modelización No Lineal")
    
    st.markdown("""
    **Insight Scouting** es una plataforma avanzada de análisis futbolístico que combina:
    - 🤖 **Machine Learning** para predicción de valores y rendimiento
    - 📊 **Analytics avanzado** con más de 22,000 jugadores evaluados
    - 🎯 **5 módulos especializados** para diferentes necesidades de scouting
    - 📈 **Visualizaciones interactivas** con radares y comparativas
    """)
    
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
    
    # Gráficos (FIX: update_layout en lugar de update_yaxis)
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
        # FIX: Usar update_layout
        fig.update_layout(yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 🎯 ¿Qué puedes hacer con Insight Scouting?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='module-card'>
        <h3>🔍 Búsqueda Directa</h3>
        <p>Busca cualquier jugador por nombre y accede a su perfil completo: stats, valoración, radar de rendimiento y más.</p>
        <ul>
        <li>22,000+ jugadores indexados</li>
        <li>Datos de 8 temporadas</li>
        <li>Fotos y información detallada</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>🔄 Encuentra Reemplazos</h3>
        <p>Algoritmo de similitud basado en ML que encuentra jugadores con perfil estadístico equivalente.</p>
        <ul>
        <li>Similitud por percentiles ponderados</li>
        <li>Filtros por presupuesto</li>
        <li>Mismo arquetipo táctico</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>⭐ Detecta Wonderkids</h3>
        <p>Identifica jóvenes talentos con alto rendimiento y potencial de revalorización.</p>
        <ul>
        <li>Score ajustado por edad</li>
        <li>Filtros personalizables</li>
        <li>Inversión en futuro</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='module-card'>
        <h3>💰 Best by Budget</h3>
        <p>Maximiza calidad dentro de tu presupuesto. Encuentra los mejores jugadores según performance score.</p>
        <ul>
        <li>Ordenamiento personalizable</li>
        <li>Filtros por liga y edad</li>
        <li>Exportación a CSV</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>💎 Jugadores Infravalorados</h3>
        <p>Detecta oportunidades de mercado: jugadores cuyo rendimiento supera su valoración actual (Gap Ratio > 1.5).</p>
        <ul>
        <li>Predicción ML de valor real</li>
        <li>ROI potencial alto</li>
        <li>Gangas confirmadas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='module-card'>
        <h3>🔄 Flip Opportunities</h3>
        <p>Estrategia de trading: compra barato, revende caro. Combina infravaloración + edad óptima.</p>
        <ul>
        <li>Edad configurable</li>
        <li>Potencial de plusvalía</li>
        <li>Análisis de mercado</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Funcionalidades Extra")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**📊 Comparador**\nSelecciona hasta 3 jugadores y compara sus stats lado a lado con radar interactivo")
    with col_b:
        st.success("**⭐ Watchlist**\nGuarda tus jugadores favoritos y tenlos siempre a mano")
    with col_c:
        st.warning("**📜 Historial**\nRevisa tus últimas 50 búsquedas")

# MÓDULO COMPARADOR (FIX: casting de player_id)
elif "Comparador" in module:
    st.title("📊 Comparador de Jugadores")
    
    if len(st.session_state.compare_players) == 0:
        st.info("💡 Selecciona jugadores desde cualquier búsqueda")
    else:
        st.success(f"✅ {len(st.session_state.compare_players)} seleccionados")
        
        if st.button("🗑️ Limpiar"):
            st.session_state.compare_players = []
            st.rerun()
        
        compare_data = []
        for pid in st.session_state.compare_players:
            try:
                player_id, season = pid.split('_')
                # FIX: Manejar player_id como string
                player = df[(df['player_id'].astype(str) == player_id) & (df['season'] == season)]
                if len(player) > 0:
                    compare_data.append(player.iloc[0])
            except:
                continue
        
        if len(compare_data) > 0:
            cols = st.columns(len(compare_data))
            for idx, player in enumerate(compare_data):
                with cols[idx]:
                    if pd.notna(player.get('player_image_url')):
                        st.image(player['player_image_url'], use_column_width=True)
                    st.markdown(f"### {player['Player']}")
                    if pd.notna(player.get('performance_score')):
                        st.metric("Score", f"{player['performance_score']:.1f}")
                    st.metric("Valor", f"€{player['market_value_current']/1e6:.1f}M")
                    st.metric("Edad", f"{int(player['Age'])}")
            
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
        else:
            st.warning("Error al cargar jugadores")

# MÓDULO WATCHLIST (FIX: casting)
elif "Watchlist" in module:
    st.title("⭐ Watchlist")
    
    if len(st.session_state.watchlist) == 0:
        st.info("💡 Tu watchlist está vacía")
    else:
        st.success(f"✅ {len(st.session_state.watchlist)} guardados")
        
        if st.button("🗑️ Limpiar"):
            st.session_state.watchlist = []
            st.rerun()
        
        watchlist_players = []
        for pid in st.session_state.watchlist:
            try:
                player_id, season = pid.split('_')
                # FIX: Casting
                player = df[(df['player_id'].astype(str) == player_id) & (df['season'] == season)]
                if len(player) > 0:
                    watchlist_players.append(player.iloc[0])
            except:
                continue
        
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
        st.info("💡 Tu historial está vacío")
    else:
        st.success(f"✅ {len(st.session_state.search_history)} búsquedas registradas")
        
        if st.button("🗑️ Limpiar historial"):
            st.session_state.search_history = []
            st.rerun()
        
        for idx, search in enumerate(reversed(st.session_state.search_history)):
            with st.expander(f"🔍 {search['type']} - {search['timestamp']}", expanded=(idx==0)):
                for key, value in search['params'].items():
                    st.write(f"**{key}**: {value}")

# MÓDULO BÚSQUEDA
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
        add_to_history('Búsqueda directa', {'nombre': search_name, 'temporada': search_season})
        
        df_search = df[df['Player'].str.contains(search_name, case=False, na=False)]
        
        if len(search_liga) > 0:
            df_search = df_search[df_search['domestic_league'].isin(search_liga)]
        
        if len(df_search) == 0:
            st.warning("No se encontraron jugadores")
        else:
            st.success(f"✅ {len(df_search)} resultados")
            
            for season in sorted(df_search['season'].unique(), reverse=True):
                season_data = df_search[df_search['season'] == season]
                if len(season_data) > 0:
                    st.markdown(f"### 📅 Temporada {season}")
                    for idx, player in season_data.iterrows():
                        with st.expander(f"⚽ {player['Player']}", expanded=(season==search_season and len(season_data)==1)):
                            show_player_profile(player)

# MÓDULO 1: BEST BY BUDGET
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
        add_to_history('Best by Budget', {'presupuesto': f'€{budget/1e6:.0f}M', 'posición': position, 'temporada': season})
        
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
            cols_to_export = [col for col in ['Player', 'Age', 'archetype', 'performance_score', 'market_value_current', 'predicted_transfer_fee'] if col in df_result.columns]
            export_to_csv(df_result[cols_to_export])
        
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
                        if pd.notna(player.get('performance_score')):
                            st.markdown(f"<div class='score-badge'>{player['performance_score']:.1f}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='price-tag'>€{player['market_value_current']/1e6:.1f}M</div>", unsafe_allow_html=True)
                        if st.button("Ver", key=f"btn_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()

# MÓDULO 3: UNDERVALUED (sin mensaje info)
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
        add_to_history('Undervalued', {'gap_mínimo': min_gap, 'posición': position, 'temporada': season})
        
        df_search = df[
            (df['season'] == season) &
            (df['gap_ratio'] >= min_gap) &
            (df['performance_score'] >= 60) &
            (df['total_90s'] >= 15) &
            (df['market_value_current'] >= 3_000_000)
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
            cols_to_export = [col for col in ['Player', 'Age', 'archetype', 'performance_score', 'market_value_current', 'gap_ratio'] if col in df_result.columns]
            export_to_csv(df_result[cols_to_export], "undervalued.csv")
        
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

# MÓDULO 4: WONDERKIDS (IMPLEMENTADO)
elif "Wonderkids" in module:
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
        add_to_history('Wonderkids', {'edad_máxima': max_age, 'presupuesto': f'€{budget/1e6:.0f}M', 'posición': position})
        
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
        
        col_exp1, col_exp2 = st.columns([3, 1])
        with col_exp1:
            st.success(f"✅ {len(df_result)} wonderkids")
        with col_exp2:
            cols_to_export = [col for col in ['Player', 'Age', 'performance_score', 'wk_score', 'market_value_current'] if col in df_result.columns]
            export_to_csv(df_result[cols_to_export], "wonderkids.csv")
        
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
                        if st.button("Ver", key=f"wk_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()

# MÓDULO 5: FLIP (edad configurable)
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
        add_to_history('Flip Opportunities', {'presupuesto': f'€{budget/1e6:.0f}M', 'edad': f'{min_age}-{max_age}', 'posición': position})
        
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
            cols_to_export = [col for col in ['Player', 'Age', 'performance_score', 'market_value_current', 'gap_ratio'] if col in df_result.columns]
            export_to_csv(df_result[cols_to_export], "flip_opportunities.csv")
        
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

# MÓDULO 2: FIND REPLACEMENT (implementación completa)
elif "Find Replacement" in module:
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
            add_to_history('Find Replacement', {'jugador': player_name, 'presupuesto': f'€{budget:.0f}M'})
            
            ref_candidates = df[df['Player'].str.contains(player_name, case=False, na=False)]
            if len(ref_candidates) == 0:
                st.error("Jugador no encontrado")
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
            st.markdown("### 🔄 Perfil seleccionado:")
            show_player_profile(player)
            
            if st.button("⬅️ Volver"):
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
                        if pd.notna(player.get('performance_score')):
                            st.markdown(f"⭐ {player['performance_score']:.1f}")
                        st.markdown(f"💰 €{player['market_value_current']/1e6:.1f}M")
                        
                        if st.button("Ver", key=f"repl_{i}_{j}"):
                            st.session_state.selected_player_idx = i + j
                            st.rerun()
