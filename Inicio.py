import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.title("Demo de TF-IDF con Preguntas y Respuestas")
st.write(
    "Ingresa varios documentos (uno por línea) y una pregunta, todo en **inglés**. "
    "La app normaliza y aplica *stemming* para que formas como *playing/play* se traten como equivalentes. "
    "Calcula TF-IDF, encuentra el documento más relevante para tu pregunta y muestra las similitudes."
)

text_input = st.text_area(
    "Documents (one per line, in English):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)
question = st.text_input("Question (in English):", "Who is playing?")

stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("Calcular TF-IDF y buscar respuesta"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )
        X = vectorizer.fit_transform(documents)
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.write("### Matriz TF-IDF (stems)")
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.write("### Pregunta y respuesta")
        st.write(f"**Tu pregunta:** {question}")
        st.write(f"**Documento más relevante (Doc {best_idx+1}):** {best_doc}")
        st.write(f"**Puntaje de similitud:** {best_score:.3f}")

        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        }).sort_values("Similitud", ascending=False)
        st.write("### Puntajes de similitud (ordenados)")
        st.dataframe(sim_df, use_container_width=True)

        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.write("### Stems de la pregunta presentes en el documento elegido:", matched)

        st.write("### Tokens de la pregunta (preprocesados)")
        st.write(q_stems)
