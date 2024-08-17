import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from wordcloud import WordCloud
import plotly.express as px

import seaborn as sns
import matplotlib.pyplot as plt
import cohere
from dotenv import load_dotenv
from os import environ

load_dotenv()
# Configura tu clave de API de Cohere
co = cohere.Client(environ.get("COHERE_API_KEY"))

# Función para hacer la solicitud a la API de TikTok y obtener los comentarios
def obtener_comentarios(url_tiktok):
    api_url = 'https://tiktok-scraper7.p.rapidapi.com/comment/list'
    headers = {
        'x-rapidapi-host': 'tiktok-scraper7.p.rapidapi.com',
        'x-rapidapi-key': f'{environ.get("RAPIDAPI_KEY")}'
    }
    params = {
        'url': url_tiktok,
        'count': 35,
        'cursor': 0
    }
    try:
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data["code"] == 0:
            return data["data"]["comments"]
        else:
            st.error("No se pudieron obtener los comentarios. Verifica el enlace o intenta nuevamente.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error en la solicitud: {e}")
        return None

# Función para analizar sentimientos, generar embeddings y crear visualizaciones
def analizar_sentimientos_y_visualizar(comentarios):
    texts = [comentario["text"] for comentario in comentarios]

    #texts append custom value
    # texts.append("Bolivia")
    # texts.append("Colombia")
    # texts.append("Bolivianos")

    # Realizar análisis de sentimientos
    examples = [
        cohere.ClassifyExample(text="Estoy muy feliz", label="Positive"),
        cohere.ClassifyExample(text="Estoy emocionado por lo que viene", label="Positive"),
        cohere.ClassifyExample(text="¡Qué día tan maravilloso!", label="Positive"),
        cohere.ClassifyExample(text="Estoy satisfecho con el resultado", label="Positive"),
        cohere.ClassifyExample(text="Estoy un poco nervioso, pero optimista", label="Neutral"),
        cohere.ClassifyExample(text="Es un día normal, nada especial", label="Neutral"),
        cohere.ClassifyExample(text="No tengo una opinión clara", label="Neutral"),
        cohere.ClassifyExample(text="Me siento tranquilo", label="Neutral"),
        cohere.ClassifyExample(text="Esto me causa preocupación", label="Negative"),
        cohere.ClassifyExample(text="Estoy frustrado con la situación", label="Negative"),
        cohere.ClassifyExample(text="Tengo miedo de lo que pueda pasar", label="Negative"),
        cohere.ClassifyExample(text="Me siento decepcionado por la calidad", label="Negative")
    ]
    
    response = co.classify(inputs=texts, model='embed-multilingual-v2.0', examples=examples)

    categories = [classification.prediction for classification in response.classifications]
    scores = [classification.confidence for classification in response.classifications]

    # Crear un DataFrame para la clasificación de sentimientos
    df = pd.DataFrame({
        'Text': texts,
        'Sentiment': categories,
        'Score': scores
    })

    # Mostrar la tabla categorizada
    
    st.subheader("Clasificación de Sentimientos:")
    #last 35 comments
    st.write("_(Últimos 35 comentarios)_")
    st.write(df)

    # Obtener embeddings de los textos (después de la clasificación)
    embedding_response = co.embed(model='embed-english-light-v3.0', texts=texts , input_type='classification')
    embeddings = np.array(embedding_response.embeddings)

    # Clustering de embeddings para agrupar textos similares
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
    labels = clustering_model.fit_predict(embeddings)

    # Crear un diccionario para normalizar los textos agrupados
    normalized_texts = {}
    for i, label in enumerate(labels):
        normalized_texts.setdefault(label, []).append(texts[i])

    # Combinar los textos normalizados
    normalized_text_combined = " ".join([" ".join(group) for group in normalized_texts.values()])

    # Crear una nube de palabras normalizada
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(normalized_text_combined)
    
    # Mostrar la nube de palabras en Streamlit
    st.image(wordcloud.to_array(), use_column_width=True)

    # Gráfico de dispersión (x, y) utilizando los embeddings
    df_embeddings = pd.DataFrame(embeddings, columns=[f'Dim_{i+1}' for i in range(embeddings.shape[1])])
    df_embeddings['Sentiment'] = categories
    df_embeddings['Comment'] = texts
    
    # Seleccionar solo dos dimensiones para graficar
    fig_scatter = px.scatter(df_embeddings, x='Dim_1', y='Dim_2', color='Sentiment', title='Embeddings de Textos por Sentimiento' , hover_data={'Comment': True})
    st.plotly_chart(fig_scatter)

    # Crear un mapa de calor basado en las dos primeras dimensiones de los embeddings
    heatmap_fig = px.density_heatmap(df_embeddings, x='Dim_1', y='Dim_2', title='Mapa de Calor de los Embeddings')
    st.plotly_chart(heatmap_fig)

    # Combinar el gráfico de dispersión con el mapa de calor
    fig_combined = px.density_heatmap(df_embeddings, x='Dim_1', y='Dim_2', title='Embeddings de Textos y Mapa de Calor')
    fig_combined.add_trace(
        px.scatter(df_embeddings, x='Dim_1', y='Dim_2', color='Sentiment', hover_data={'Comment': True}).data[0]
    )
    
    # Mostrar el gráfico combinado en Streamlit
    st.plotly_chart(fig_combined)
    

    

st.markdown("""
### Código Fuente
Puedes ver y modificar el código en el siguiente repositorio de GitHub: [Repositorio de GitHub](https://github.com/MrDnck/Tiktok-Analysis)

### Autor
Cristian Catari
            
Celular: +591 70562921  
Correo electrónico: cristian.catari.ma@gmail.com
""")
st.video('https://youtu.be/N7psa-BSxsU')

# Streamlit app
st.title('TikTok Comment Scraper & Sentiment Analysis')

# Entrada de usuario para el link de TikTok
url_input = st.text_input('Inserta el enlace del video de TikTok:')

if url_input:
    comentarios = obtener_comentarios(url_input)
    
    if comentarios:
        analizar_sentimientos_y_visualizar(comentarios)
