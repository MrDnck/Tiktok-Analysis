# TikTok Comment Scraper & Sentiment Analysis

Esta aplicación de Streamlit permite extraer comentarios (maximo 35) de videos de TikTok y realizar un análisis de sentimientos sobre esos comentarios. Utiliza técnicas de clustering y visualización para proporcionar una comprensión más profunda de los sentimientos y patrones en los comentarios.

## Demo

Puedes ver una demo de la aplicación en este enlace: [Demo](https://tiktok-analysis.streamlit.app/)

## Descripción

La aplicación muestra:
- Extracción automática de comentarios de un video de TikTok utilizando una API.
- Clasificación de sentimientos (positivo, neutral, negativo) para cada comentario.
- Visualización de una nube de palabras basada en los comentarios agrupados.
- Gráfico de dispersión de los embeddings generados a partir de los comentarios, categorizados por sentimiento.
- Mapa de calor que muestra la densidad de los puntos en el espacio de los embeddings para identificar patrones de agrupamiento.

## Instalación

1. Instala las dependencias del proyecto:

```bash
pip install -r requirements.txt
```
2. Ejecuta el script de Python:

```bash
streamlit run app.py
```