#import logging
#logging.getLogger('streamlit').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Configuração da página
st.set_page_config(page_title="Dashboard Netflix", layout="wide")

# Função para carregar e tratar os dados
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")

    # Remover colunas desnecessárias
    colunas_remover = ["director", "cast", "description", "rating"]
    df.drop(columns=colunas_remover, errors="ignore", inplace=True)

    # Remover registros sem país
    df.dropna(subset=["country"], inplace=True)

    # Renomear colunas
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Corrigir datas
    df['date_added'] = pd.to_datetime(df['date_added'].astype(str).str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    if df['year_added'].isnull().any():
        df['year_added'] = df['year_added'].fillna(df['year_added'].mode()[0])
    df['year_added'] = df['year_added'].astype('Int64')
    df['month_added'] = df['date_added'].dt.month

    df.dropna(subset=["date_added", "duration"], inplace=True)
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df['duracao_minutos'] = df['duration'].str.extract(r'(\d+)').astype(float)

    # Para análise de múltiplos países e gêneros
    df['country'] = df['country'].astype(str).str.strip().str.split(', ')
    df['listed_in'] = df['listed_in'].astype(str).str.strip().str.split(', ')

    return df

# Carregar dados
df = load_data()

# aqui ta o sidebar (aquela coluna lateral)
with st.sidebar:
        st.title("Dashboard Netflix")
        st.write("Análise de dados dos títulos disponíveis na Netflix")
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Netflix_logo.svg", width=200)

        tipo = st.radio("Selecione o tipo de título:", options=["Filme", "Série"], index=0)
        paises = st.multiselect(
            "Selecione os países:",
            options=pd.read_csv("netflix_titles.csv")['country'].dropna().unique().tolist(),
            default=["United States", "Brazil", "India"],
            help="Selecione um ou mais países para filtrar os dados"
        )
        anos = st.slider(
            "Selecione o intervalo de anos:",
            min_value=1950,
            max_value=2021,
            value=(2000, 2021),
            step=1,
            help="Selecione o intervalo de anos para filtrar os dados"
        )
        st.write("### Sobre o Dashboard")
        st.write('Criado com o objetivo de análise sobre popularidade e tendências dos títulos disponíveis na Netflix.')

        st.write("### Sobre os Dados") 
        st.write("Os dados foram extraídos de um dataset público do Kaggle, contendo informações sobre títulos disponíveis na Netflix, incluindo tipo, país, ano de lançamento e gênero.")
"""        
df_filtrado = df[df['type'] == tipo]
if paises:
    df_filtrado = df_filtrado[df_filtrado['country'].apply(lambda x: any(pais in x for pais in paises))] 
df_filtrado = df_filtrado[
    (df_filtrado['year_added'] >= anos[0]) & 
    (df_filtrado['year_added'] <= anos[1])
    ]
"""
# Título
st.title("📊 Análise dos Dados Netflix")
st.write(f"✅ Dados carregados com sucesso!")

# Filtro
tipo = st.selectbox("🎬 Selecione o tipo de título:", options=df['type'].unique())
df_filtrado = df[df['type'] == tipo]

# Coluna categórica
coluna_selecionada = st.selectbox(
    "📌 Selecione uma coluna categórica para análise:",
    options=['country', 'listed_in', 'year_added']
)

# Explodir se necessário
if coluna_selecionada in ['country', 'listed_in']:
    df_exp = df_filtrado.explode(coluna_selecionada)
    contagem = df_exp[coluna_selecionada].value_counts().head(20)
else:
    contagem = df_filtrado[coluna_selecionada].value_counts().head(20)

# Gráfico interativo
fig = px.bar(
    x=contagem.index,
    y=contagem.values,
    labels={'x': coluna_selecionada.capitalize(), 'y': 'Quantidade'},
    title=f"Top valores em '{coluna_selecionada}' para '{tipo}'",
    color=contagem.values,
    color_continuous_scale='reds'
)
st.plotly_chart(fig, use_container_width=True)


# ================== ABAS COM GRÁFICOS INTERATIVOS PLOTLY ==================
abas = st.tabs(["🎞️ Tipos de Título", "🍿 Gêneros Populares", "🌍 Gêneros por País", "📅 Evolução de Lançamentos ao Longo dos Anos", "🌍 Distribuição de Títulos por País"])

with abas[0]:
    st.subheader("🎞️ Proporção de Filmes vs Séries")
    type_counts = df['type'].value_counts()
    labels = type_counts.index.tolist()
    values = type_counts.values
    porcentagens = np.round((values / np.sum(values)) * 100, 1)

    # Criar DataFrame para Plotly
    df_type = pd.DataFrame({'Tipo': labels, 'Quantidade': values, 'Porcentagem': porcentagens})

    fig = px.bar(
        df_type,
        x='Tipo',
        y='Quantidade',
        color='Quantidade',
        color_continuous_scale=['#B81D24', '#E50914'],
        text=df_type['Porcentagem'].astype(str) + '%',
        title='Número de Filmes vs Shows de TV'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        plot_bgcolor='#2c2c2c',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        yaxis=dict(range=[0, max(values)*1.2]),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

with abas[1]:
    st.subheader("🍿 Top 10 Gêneros Mais Populares")
    generos_explodidos = df.explode('listed_in')["listed_in"]
    contagem_generos = generos_explodidos.value_counts().head(10)
    generos_df = pd.DataFrame({'Gênero': contagem_generos.index, 'Quantidade': contagem_generos.values})

    fig = px.bar(
        generos_df,
        x='Quantidade',
        y='Gênero',
        orientation='h',
        color='Quantidade',
        color_continuous_scale='reds',
        text=generos_df['Quantidade'],
        title='Top 10 Gêneros Mais Populares na Netflix'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='#2c2c2c',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

with abas[2]:
    st.subheader("🌍 Distribuição de Gêneros por País (Top 10)")
    genre_analysis = df.copy()
    genre_analysis = genre_analysis.explode('listed_in')
    genre_analysis = genre_analysis.explode('country')

    top_genres = genre_analysis['listed_in'].value_counts().head(10).index.tolist()
    top_countries = genre_analysis['country'].value_counts().head(10).index.tolist()

    filtro = genre_analysis[
        genre_analysis['listed_in'].isin(top_genres) &
        genre_analysis['country'].isin(top_countries)
    ]
    pivot = pd.crosstab(
        index=filtro['country'],
        columns=filtro['listed_in'],
        values=filtro['show_id'],
        aggfunc='count'
    ).fillna(0)

    pivot = pivot[top_genres]
    pivot_normalized = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Resetar índice para plotar com Plotly
    df_heatmap = pivot_normalized.reset_index().melt(id_vars='country', var_name='Gênero', value_name='Porcentagem')

    fig = px.density_heatmap(
        df_heatmap,
        x='Gênero',
        y='country',
        z='Porcentagem',
        color_continuous_scale='YlGnBu',
        text_auto='.1f',
        title='Proporção de Gêneros por País (Top 10 Global)'
    )
    fig.update_layout(
        plot_bgcolor='#2c2c2c',
        paper_bgcolor='#1e1e1e',
        font_color='white',
        xaxis_tickangle=45,
        yaxis_title='País',
        xaxis_title='Gênero'
    )
    st.plotly_chart(fig, use_container_width=True)
with abas[3]:
    st.subheader("📅 Evolução de Lançamentos ao Longo dos Anos")

    df_ano_tipo = df.groupby(['release_year', 'type']).size().reset_index(name='Contagem')
    fig = px.line(
        df_ano_tipo,
        x='release_year',
        y='Contagem',
        color='type',
        markers=True,
        title="Número de Títulos Lançados por Ano",
        labels={'release_year': 'Ano de Lançamento'}
    )
    fig.update_layout(paper_bgcolor='#1e1e1e', plot_bgcolor='#2c2c2c', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
with abas[4]:
    st.subheader("⌛ Duração Média por Tipo")
    duracao_tipo = df.groupby('type')['duracao_minutos'].mean().reset_index()
    fig = px.pie(
        duracao_tipo,
        names='type',
        values='duracao_minutos',
        title='Duração Média (em minutos)',
        color_discrete_sequence=['#E50914', '#B81D24']
    )
    fig.update_layout(paper_bgcolor='#1e1e1e', font_color='white')
    st.plotly_chart(fig, use_container_width=True)


    df_country = df.explode('country')
    top_paises = df_country['country'].value_counts().head(20).reset_index()
    top_paises.columns = ['País', 'Total']
