import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
import numpy as np

# Download required NLTK data if needed
try:
    import nltk
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="Demo TF-IDF Español", page_icon="🔍", layout="wide")

st.title("🔍 Demo de TF-IDF en Español con Preguntas y Respuestas")

st.markdown("""
### ¿Cómo funciona?
Cada línea se trata como un **documento** (puede ser una frase, un párrafo o un texto más largo).  
La aplicación utiliza:
- **TF-IDF**: Para calcular la importancia de cada palabra en cada documento
- **Stemming**: Para que palabras como *jugando* y *juegan* se consideren equivalentes
- **Similitud del coseno**: Para encontrar el documento más relevante a tu pregunta
""")

# Sidebar con configuraciones
st.sidebar.header("⚙️ Configuración")
min_df = st.sidebar.slider("Frecuencia mínima de términos", 1, 5, 1, 
                          help="Ignora términos que aparecen en menos de N documentos")
max_features = st.sidebar.selectbox("Máximo número de características", 
                                   [None, 100, 500, 1000], index=0)
similarity_threshold = st.sidebar.slider("Umbral de similitud", 0.0, 1.0, 0.1, 0.01,
                                        help="Documentos con similitud menor a este valor se marcarán como poco relevantes")

# Dos columnas para la interfaz principal
col1, col2 = st.columns([2, 1])

with col1:
    # Ejemplo inicial mejorado
    default_text = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

    text_input = st.text_area(
        "📝 Escribe tus documentos (uno por línea):",
        default_text,
        height=200,
        help="Cada línea será tratada como un documento separado"
    )

    question = st.text_input(
        "❓ Escribe una pregunta:", 
        "¿Quién está jugando?",
        help="La aplicación buscará el documento más relevante para responder tu pregunta"
    )

with col2:
    st.markdown("### 💡 Ejemplos de preguntas:")
    st.markdown("""
    - ¿Quién está jugando?
    - ¿Qué animal hace ruido?
    - ¿Dónde corren los niños?
    - ¿Cuándo cantan los pájaros?
    - ¿Qué hace el gato?
    """)

# Inicializar stemmer para español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text: str):
    """
    Tokeniza y aplica stemming a un texto en español.
    """
    # Pasar a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfabéticos (mantener caracteres españoles)
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar (palabras con longitud > 1)
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def highlight_matches(text: str, stems: list) -> str:
    """
    Resalta las palabras que coinciden con los stems en el texto.
    """
    words = text.split()
    highlighted = []
    for word in words:
        word_stem = stemmer.stem(re.sub(r'[^a-záéíóúüñ]', '', word.lower()))
        if word_stem in stems:
            highlighted.append(f"**{word}**")
        else:
            highlighted.append(word)
    return " ".join(highlighted)

# Botón principal
if st.button("🔍 Analizar documentos y buscar respuesta", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Por favor, ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Por favor, escribe una pregunta.")
    else:
        with st.spinner("Procesando documentos..."):
            # Configurar vectorizador
            vectorizer_params = {
                'tokenizer': tokenize_and_stem,
                # 'stop_words': 'spanish',  # Removido - manejamos stopwords en tokenize_and_stem
                'min_df': min_df,
            }
            if max_features:
                vectorizer_params['max_features'] = max_features
                
            vectorizer = TfidfVectorizer(**vectorizer_params)
            
            try:
                # Ajustar con documentos
                X = vectorizer.fit_transform(documents)
                
                # Vector de la pregunta
                question_vec = vectorizer.transform([question])
                
                # Similitud coseno
                similarities = cosine_similarity(question_vec, X).flatten()
                
                # Resultados
                st.success("✅ Análisis completado!")
                
                # Crear tabs para organizar mejor la información
                tab1, tab2, tab3 = st.tabs(["🎯 Resultado Principal", "📊 Análisis Detallado", "🔢 Matriz TF-IDF"])
                
                with tab1:
                    # Documento más parecido
                    best_idx = similarities.argmax()
                    best_doc = documents[best_idx]
                    best_score = similarities[best_idx]
                    
                    st.markdown("### 🎯 Resultado de búsqueda")
                    
                    col_q, col_a = st.columns(2)
                    with col_q:
                        st.markdown("**❓ Tu pregunta:**")
                        st.info(question)
                    
                    with col_a:
                        st.markdown("**💡 Documento más relevante:**")
                        if best_score >= similarity_threshold:
                            # Resaltar palabras que coinciden
                            q_stems = tokenize_and_stem(question)
                            vocab = vectorizer.get_feature_names_out()
                            matched_stems = [s for s in q_stems if s in vocab]
                            highlighted_doc = highlight_matches(best_doc, matched_stems)
                            st.success(highlighted_doc)
                            st.caption(f"📈 Puntaje de similitud: {best_score:.3f}")
                        else:
                            st.warning(f"⚠️ {best_doc}")
                            st.caption(f"📉 Similitud baja: {best_score:.3f} (< {similarity_threshold})")
                
                with tab2:
                    # Mostrar todas las similitudes
                    sim_df = pd.DataFrame({
                        "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                        "Texto": documents,
                        "Similitud": similarities
                    })
                    sim_df = sim_df.sort_values("Similitud", ascending=False)
                    
                    st.markdown("### 📊 Ranking de documentos")
                    
                    # Agregar colores basados en similitud
                    def color_similarity(val):
                        if val >= 0.5:
                            return 'background-color: #d4edda'  # Verde claro
                        elif val >= 0.2:
                            return 'background-color: #fff3cd'  # Amarillo claro
                        else:
                            return 'background-color: #f8d7da'  # Rojo claro
                    
                    styled_df = sim_df.style.applymap(color_similarity, subset=['Similitud'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Mostrar coincidencias de stems
                    vocab = vectorizer.get_feature_names_out()
                    q_stems = tokenize_and_stem(question)
                    matched = [s for s in q_stems if s in vocab and 
                              sim_df.iloc[0]['Similitud'] > 0]  # Del documento con mayor similitud
                    
                    if matched:
                        st.markdown("### 🔤 Términos coincidentes (después de stemming)")
                        st.info(f"Términos encontrados: {', '.join(matched)}")
                    else:
                        st.warning("No se encontraron términos coincidentes después del procesamiento")
                
                with tab3:
                    # Mostrar matriz TF-IDF
                    df_tfidf = pd.DataFrame(
                        X.toarray(),
                        columns=vectorizer.get_feature_names_out(),
                        index=[f"Doc {i+1}" for i in range(len(documents))]
                    )
                    
                    st.markdown("### 🔢 Matriz TF-IDF")
                    st.caption("Valores mayores indican mayor importancia del término en el documento")
                    
                    # Filtrar columnas con valores > 0 para mejor visualización
                    non_zero_cols = df_tfidf.columns[df_tfidf.sum() > 0]
                    if len(non_zero_cols) > 20:
                        st.warning(f"Mostrando solo las primeras 20 columnas de {len(non_zero_cols)} términos")
                        display_df = df_tfidf[non_zero_cols[:20]]
                    else:
                        display_df = df_tfidf[non_zero_cols]
                    
                    st.dataframe(display_df.round(3), use_container_width=True)
                    
                    # Estadísticas del vocabulario
                    st.markdown("### 📈 Estadísticas del vocabulario")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        st.metric("Documentos", len(documents))
                    with col_stats2:
                        st.metric("Términos únicos", len(vectorizer.get_feature_names_out()))
                    with col_stats3:
                        st.metric("Similitud máxima", f"{similarities.max():.3f}")
            
            except ValueError as e:
                st.error(f"❌ Error en el procesamiento: {str(e)}")
                st.info("💡 Intenta agregar más documentos o usar palabras más variadas")

# Información adicional en la sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Información técnica")
st.sidebar.markdown("""
**TF-IDF** significa *Term Frequency - Inverse Document Frequency* 
y mide la importancia de una palabra en un documento dentro de una colección.

**Stemming** reduce las palabras a su raíz, por ejemplo:
- jugando → jug
- juegan → jug  
- jugador → jug

**Similitud del coseno** mide qué tan similares son dos vectores, 
donde 1 = idénticos y 0 = completamente diferentes.
""")

st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado  usando Streamlit y scikit-learn")
