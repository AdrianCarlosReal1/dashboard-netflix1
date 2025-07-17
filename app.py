import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# ========== CONFIGURAÇÃO DA PÁGINA ========== #
st.set_page_config(page_title="Dashboard Netflix", layout="wide")

# ========== FUNÇÃO DE TRATAMENTO DE DADOS ========== #
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")

    # Remover colunas desnecessárias
    df.drop(columns=["director", "cast", "description", "rating"], errors="ignore", inplace=True)

    # Remover registros sem país
    df.dropna(subset=["country"], inplace=True)

    # Padronizar nomes de colunas
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Tratar datas
    df['date_added'] = pd.to_datetime(df['date_added'].astype(str).str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['year_added'] = df['year_added'].fillna(df['year_added'].mode()[0]).astype('Int64')
    df['month_added'] = df['date_added'].dt.month

    # Remover registros com datas ou duração ausente
    df.dropna(subset=["date_added", "duration"], inplace=True)

    # Garantir tipos numéricos
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df['duracao_minutos'] = df['duration'].str.extract(r'(\d+)').astype(float)

    # Separar listas
    df['country'] = df['country'].astype(str).str.strip().str.split(', ')
    df['listed_in'] = df['listed_in'].astype(str).str.strip().str.split(', ')

    # Explodir países para filtragem correta
    df = df.explode('country')

    return df

# ========== CARREGAMENTO DOS DADOS ========== #
df = load_data()
df_treino = df[~df['year_added'].isin([2019])].copy()

# ========== SIDEBAR ========== #
with st.sidebar:
    st.title("🎬 Dashboard Netflix")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Netflix_logo.svg", width=200)

    tipo = st.radio("🎞️ Tipo de título:", options=["Movie", "TV Show"], index=0)
    paises = st.multiselect(
        "🌎 Filtrar por país:",
        options=sorted(df['country'].dropna().unique().tolist()),
        default=["United States", "Brazil", "India"]
    )

    st.markdown("---")

    st.markdown("**🔍 Sobre a Análise**")
    st.markdown("""
    Este dashboard explora os dados de títulos disponíveis na plataforma **Netflix** — incluindo informações sobre filmes, séries, gêneros, países e evolução ao longo do tempo.
    """)

    st.markdown("**📂 Origem do Dataset**")
    st.markdown("""
    Os dados foram extraídos de um [dataset público do Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows) contendo metadados dos títulos disponíveis na Netflix até 2021.
    """)

    st.markdown("**💻 Repositório no GitHub**")
    st.markdown("""
    Acesse o projeto completo com código e documentação no [GitHub do projeto](https://github.com/AdrianCarlosReal1/dashboard-netflix1).
    """)

    st.markdown("---")

# ========== APLICAR FILTROS ========== #
df_filtrado = df[df['type'] == tipo]

if paises:
    df_filtrado = df_filtrado[df_filtrado['country'].isin(paises)]


# ========== VISUALIZAÇÃO PRINCIPAL ========== #

st.title("📊 Análise dos Dados Netflix")
st.write("✅ Dados carregados e tratados com sucesso!")

# 🏆 Painel de Destaques
st.markdown("### 🏆 Destaques da Netflix")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
# Título mais antigo
titulo_mais_antigo = df.loc[df['release_year'] == df['release_year'].min(), 'title'].iloc[0]

# Título mais recente
titulo_mais_novo = df.loc[df['release_year'] == df['release_year'].max(), 'title'].iloc[0]

# Criar nova coluna com minutagem para filmes
df['duracao_minutos'] = df['duration'].apply(
    lambda x: int(x.replace(' min', '')) if isinstance(x, str) and 'min' in x else None
)

# Média de duração dos filmes
media_filme = df[df['type'] == 'Movie']['duracao_minutos'].mean()

# Totais
total_titulos = df.shape[0]
total_filmes = df[df['type'] == 'Movie'].shape[0]
total_series = df[df['type'] == 'TV Show'].shape[0]

# Gêneros por título (média)
media_generos = df['listed_in'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()

# Quantidade de países únicos
qtd_paises = df['country'].explode().nunique()

# Métricas visuais
col1.metric("🎬 Título Mais Antigo", titulo_mais_antigo, f"Ano: {int(df['release_year'].min())}")
col2.metric("🆕 Título Mais Recente", titulo_mais_novo, f"Ano: {int(df['release_year'].max())}")
col3.metric("📦 Total de Títulos", f"{total_titulos:,}", f"{total_filmes} filmes / {total_series} séries")
col4.metric("⏱️ Duração Média (Filmes)", f"{media_filme:.1f} min")
col5.metric("🍿 Gêneros por Título (média)", f"{media_generos:.1f}")
col6.metric("🌍 Países Representados", f"{qtd_paises} países")

coluna_selecionada = st.selectbox(
    "📌 Selecione uma coluna categórica para análise:",
    options=['country', 'listed_in', 'year_added']
)

# Explodir se necessário
if coluna_selecionada in ['country', 'listed_in']:
    df_filtrado = df_filtrado.explode(coluna_selecionada)

contagem = df_filtrado[coluna_selecionada].value_counts().head(20)

if contagem.empty:
    st.warning(f"🔍 Nenhum dado encontrado para a combinação selecionada: tipo '{tipo}' e coluna '{coluna_selecionada}'.")
else:
    fig = px.bar(
        x=contagem.index,
        y=contagem.values,
        labels={'x': coluna_selecionada.capitalize(), 'y': 'Quantidade'},
        title=f"Top valores em '{coluna_selecionada}' para '{tipo}'",
        color=contagem.values,
        color_continuous_scale='reds'
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== ABAS DE ANÁLISE ========== #
abas = st.tabs([
    "🎞️ Tipos de Título",
    "🍿 Gêneros Populares",
    "🌍 Gêneros por País",
    "📅 Evolução de Lançamentos",
    "⌛ Duração Média por Tipo",
    "📥 Adições por Ano",
    "📆 Adições Mensais de Títulos à Netflix",
    "🧩 Proporção de Gêneros - Treemap",
    "🌐 Tipos de Título por País (Top 10 países)"
])

with abas[0]:
    st.subheader("🎞️ Proporção de Filmes vs Séries")
    tipo_df = df['type'].value_counts()
    porcentagens = np.round((tipo_df / tipo_df.sum()) * 100, 1)
    fig = px.bar(

        x=tipo_df.index,
        y=tipo_df.values,
        text=[f'{p}%' for p in porcentagens],
        labels={'x': 'Tipo', 'y': 'Quantidade'},
        color=tipo_df.values,
        color_continuous_scale=['#B81D24', '#E50914'],
        title='Distribuição de Filmes vs Séries'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(plot_bgcolor='#2c2c2c', paper_bgcolor='#1e1e1e', font_color='white', coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with abas[1]:
    st.subheader("🍿 Top 10 Gêneros Mais Populares")
    generos = df.explode('listed_in')['listed_in'].value_counts().head(10)
    fig = px.bar(
        x=generos.values,
        y=generos.index,
        orientation='h',
        color=generos.values,
        color_continuous_scale='reds',
        labels={'x': 'Quantidade', 'y': 'Gênero'},
        title='Gêneros Mais Frequentes'
    )
    fig.update_layout(plot_bgcolor='#2c2c2c', paper_bgcolor='#1e1e1e', font_color='white', coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with abas[2]:
    st.subheader("🌍 Gêneros por País (Top 10)")
    genero_pais = df.copy()
    genero_pais = genero_pais.explode('listed_in').explode('country')
    top_gen = genero_pais['listed_in'].value_counts().head(10).index
    top_cty = genero_pais['country'].value_counts().head(10).index
    filtro = genero_pais[(genero_pais['listed_in'].isin(top_gen)) & (genero_pais['country'].isin(top_cty))]
    crosstab = pd.crosstab(filtro['country'], filtro['listed_in'])
    norm = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    heatmap = norm.reset_index().melt(id_vars='country', var_name='Gênero', value_name='Porcentagem')

    fig = px.density_heatmap(
        heatmap,
        x='Gênero',
        y='country',
        z='Porcentagem',
        color_continuous_scale='YlGnBu',
        text_auto='.1f',
        title='Proporção de Gêneros por País'
    )
    fig.update_layout(plot_bgcolor='#2c2c2c', paper_bgcolor='#1e1e1e', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

with abas[3]:
    st.subheader("📅 Evolução de Lançamentos ao Longo dos Anos")
    linha = df.groupby(['release_year', 'type']).size().reset_index(name='Contagem')
    fig = px.line(
        linha,
        x='release_year',
        y='Contagem',
        color='type',
        markers=True,
        labels={'release_year': 'Ano', 'Contagem': 'Títulos'},
        title='Número de Títulos por Ano'
    )
    fig.update_layout(plot_bgcolor='#2c2c2c', paper_bgcolor='#1e1e1e', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

with abas[4]:
    st.subheader("⌛ Duração Média por Tipo")
    media_duracao = df.groupby('type')['duracao_minutos'].mean().reset_index()
    fig = px.pie(
        media_duracao,
        names='type',
        values='duracao_minutos',
        title='Duração Média dos Títulos (em minutos)',
        color_discrete_sequence=['#E50914', '#B81D24']
    )
    fig.update_layout(paper_bgcolor='#1e1e1e', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
with abas[5]:
    st.subheader("📥 Títulos Adicionados à Netflix por Ano")

    anos_adicao = st.slider(
        "Selecione o intervalo de anos de adição à Netflix:",
        min_value=2008,
        max_value=2021,
        value=(2010, 2021),
        step=1
    )

    df_adicao = df[
        (df['year_added'] >= anos_adicao[0]) &
        (df['year_added'] <= anos_adicao[1])
    ]

    df_adicoes_ano = df_adicao.groupby(['year_added', 'type']).size().reset_index(name='Quantidade')

    fig = px.line(
        df_adicoes_ano,
        x='year_added',
        y='Quantidade',
        color='type',
        markers=True,
        title='Evolução de Títulos Adicionados à Netflix por Ano',
        labels={'year_added': 'Ano de Adição', 'Quantidade': 'Número de Títulos'}
    )
    fig.update_layout(paper_bgcolor='#1e1e1e', plot_bgcolor='#2c2c2c', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
with abas[6]:
    st.subheader("📆 Adições Mensais de Títulos à Netflix")

    df_mes = df.dropna(subset=['month_added'])
    df_mes_grouped = df_mes.groupby(['month_added', 'type']).size().reset_index(name='Quantidade')

    fig = px.line(
        df_mes_grouped,
        x='month_added',
        y='Quantidade',
        color='type',
        markers=True,
        title='Número de Títulos Adicionados por Mês',
        labels={'month_added': 'Mês'}
    )
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(range(1,13)), ticktext=[
            'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
            'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'
        ]),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2c2c2c',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)
with abas[7]:
    st.subheader("🧩 Proporção de Gêneros - Treemap")

    genero_treemap = df.explode('listed_in')
    contagem_gen = genero_treemap['listed_in'].value_counts().head(20).reset_index()
    contagem_gen.columns = ['Gênero', 'Quantidade']

    fig = px.treemap(contagem_gen, path=['Gênero'], values='Quantidade', title='Proporção dos 20 Gêneros Mais Populares')
    fig.update_layout(paper_bgcolor='#1e1e1e', plot_bgcolor='#2c2c2c', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
with abas[8]:
    st.subheader("🌐 Tipos de Título por País (Top 10 países)")

    top_ctys = df['country'].value_counts().head(10).index
    df_tipo_pais = df[df['country'].isin(top_ctys)]
    tipo_pais = df_tipo_pais.groupby(['country', 'type']).size().reset_index(name='Contagem')

    fig = px.bar(
        tipo_pais,
        x='country',
        y='Contagem',
        color='type',
        title='Distribuição de Filmes e Séries por País',
        barmode='stack'
    )
    fig.update_layout(paper_bgcolor='#1e1e1e', plot_bgcolor='#2c2c2c', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
# ===================================================================
# SEÇÃO 2: PREVISÃO COM IA (VERSÃO FINAL COM REGRESSÃO POLINOMIAL)
# ===================================================================
st.write("---")
st.header("🤖 Análise Preditiva de Conteúdo por País (IA)")
st.markdown("""
Esta seção utiliza **Inteligência Artificial** para prever a tendência de novas adições de conteúdo. 
Você pode controlar a complexidade do modelo de previsão usando o slider de "Grau Polinomial".
**Nota Metodológica:** Os modelos são treinados ignorando os dados dos anos de 2019 e 2021, considerados atípicos.
""")

# --- Preparar dados ---
top_countries = [
    'United States', 'India', 'United Kingdom', 'Canada', 'France', 'Spain', 'Egypt', 'Mexico', 
    'Turkey', 'Japan', 'Australia', 'China', 'Germany', 'South Korea', 'Hong Kong', 
    'Indonesia', 'Philippines', 'Nigeria'
]
df_top_countries = df_treino[df_treino['country'].isin(top_countries)].copy()
df_top_countries['year_added'] = df_top_countries['year_added'].astype(int)
country_time_series = df_top_countries.groupby(['year_added', 'country', 'type']).size().reset_index(name='title_count')

# --- Interface ---
trainable_countries = [country for country in top_countries if len(country_time_series[country_time_series['country'] == country]) > 3]
selected_country = st.selectbox("Selecione um País para Análise:", options=trainable_countries)

degree = st.slider("Ajuste a Complexidade do Modelo (Grau Polinomial):", 1, 3, 2, key="degree_slider")
st.caption(f"Grau 1: Modelo Linear (reta). Graus 2 e 3: Modelos Polinomiais (curvas).")

if selected_country:
    # --- Treinar e Avaliar Modelos Dinamicamente ---
    models = {}
    metrics = {}
    for content_type in ['Movie', 'TV Show']:
        type_data = country_time_series[(country_time_series['country'] == selected_country) & (country_time_series['type'] == content_type)]
        if len(type_data) > degree:
            X_train = type_data[['year_added']]
            y_train = type_data['title_count']
            model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
            model.fit(X_train, y_train)
            models[content_type] = model
            y_pred_train = model.predict(X_train)
            r2 = r2_score(y_train, y_pred_train)
            mae = mean_absolute_error(y_train, y_pred_train)
            metrics[content_type] = {'R2': r2, 'MAE': mae}

    # --- Mostrar Métricas ---
    st.write("---")
    st.subheader(f"Avaliação do Modelo (Grau {degree}) para {selected_country}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Modelo de Filmes 🎞️")
        metrics_movie = metrics.get('Movie')
        if metrics_movie:
            st.metric(label="Qualidade (R²)", value=f"{metrics_movie['R2']:.1%}")
            st.metric(label="Erro Médio (MAE)", value=f"{metrics_movie['MAE']:.1f} títulos/ano")
        else:
            st.info("Não há dados suficientes para este modelo.")
    with col2:
        st.markdown("##### Modelo de Séries 📺")
        metrics_show = metrics.get('TV Show')
        if metrics_show:
            st.metric(label="Qualidade (R²)", value=f"{metrics_show['R2']:.1%}")
            st.metric(label="Erro Médio (MAE)", value=f"{metrics_show['MAE']:.1f} títulos/ano")
        else:
            st.info("Não há dados suficientes para este modelo.")

    # --- Previsão e Gráficos ---
    st.write("---")
    st.subheader(f"Previsão de Novas Adições para {selected_country}")
    future_year = st.slider("Selecione o Ano para Previsão:", 2022, 2030, 2022, key=f"slider_final_poly_{selected_country}")
    
    model_movie = models.get('Movie')
    model_show = models.get('TV Show')
    pred_movie, pred_show = 0, 0
    if model_movie:
        pred_movie = max(0, int(model_movie.predict(pd.DataFrame([[future_year]], columns=['year_added']))[0]))
    if model_show:
        pred_show = max(0, int(model_show.predict(pd.DataFrame([[future_year]], columns=['year_added']))[0]))

    col1_pred, col2_pred = st.columns(2)
    col1_pred.metric(label="Novos Filmes Previstos", value=pred_movie)
    col2_pred.metric(label="Novas Séries Previstas", value=pred_show)
    
    st.write("---")
    st.subheader(f"Evolução e Tendência de Conteúdo (Modelo Grau {degree})")
    fig = go.Figure()
    if model_movie:
        history_movie = country_time_series[(country_time_series['country'] == selected_country) & (country_time_series['type'] == 'Movie')]
        fig.add_trace(go.Scatter(x=history_movie['year_added'], y=history_movie['title_count'], mode='lines+markers', name='Filmes (Histórico)', line=dict(color='royalblue')))
        trend_x_range = np.arange(history_movie['year_added'].min(), future_year + 1).reshape(-1, 1)
        trend_y_m = model_movie.predict(trend_x_range)
        fig.add_trace(go.Scatter(x=trend_x_range.flatten(), y=[max(0, val) for val in trend_y_m], mode='lines', name='Filmes (Previsão)', line=dict(dash='dash', color='cyan')))
    if model_show:
        history_show = country_time_series[(country_time_series['country'] == selected_country) & (country_time_series['type'] == 'TV Show')]
        fig.add_trace(go.Scatter(x=history_show['year_added'], y=history_show['title_count'], mode='lines+markers', name='Séries (Histórico)', line=dict(color='red')))
        trend_x_range_s = np.arange(history_show['year_added'].min(), future_year + 1).reshape(-1, 1)
        trend_y_s = model_show.predict(trend_x_range_s)
        fig.add_trace(go.Scatter(x=trend_x_range_s.flatten(), y=[max(0, val) for val in trend_y_s], mode='lines', name='Séries (Previsão)', line=dict(dash='dash', color='tomato')))
    fig.update_layout(title=f"Histórico vs. Previsão para {selected_country}", xaxis_title='Ano', yaxis_title='Quantidade de Títulos Adicionados', paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', font_color='white', legend_title_text='Legenda')
    st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.subheader(f"Proporção Prevista para {future_year} em {selected_country}")
    labels = ['Filmes', 'Séries']
    values = [pred_movie, pred_show]
    pie_fig = px.pie(names=labels, values=values, title=f'Proporção de Conteúdo para {selected_country} em {future_year}', hole=0.3)
    total_preds = pred_movie + pred_show
    if total_preds == 0:
        pie_fig.add_annotation(text="0 Títulos Previstos", x=0.5, y=0.5, font_size=20, showarrow=False)
    pie_fig.update_traces(textinfo='percent+label', marker=dict(colors=['royalblue', 'red']))
    pie_fig.update_layout(paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', font_color='white')
    st.plotly_chart(pie_fig, use_container_width=True)
