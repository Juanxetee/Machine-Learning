import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect
import pickle

# Descargar recursos necesarios para nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Cargar el modelo y el vectorizador guardados
with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Función de preprocesamiento para varios idiomas
lemmatizer = WordNetLemmatizer()

def preprocess_text_multilingual(text, language):
    if language == 'spanish':
        stop_words = set(stopwords.words('spanish'))
    elif language == 'english':
        stop_words = set(stopwords.words('english'))
    else:
        # Añadir más idiomas según sea necesario
        stop_words = set()
    
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Función para predecir el sentimiento de un comentario independiente
def predict_comment(comment):
    try:
        language = detect(comment)
    except:
        language = 'unknown'
    
    if language in ['es', 'en']:  # Añadir más códigos de idiomas según sea necesario
        if language == 'es':
            language = 'spanish'
        elif language == 'en':
            language = 'english'
        preprocessed_comment = preprocess_text_multilingual(comment, language)
        vectorized_comment = vectorizer.transform([preprocessed_comment]).toarray()
        prediction = best_model.predict(vectorized_comment)
        return 'Positivo' if prediction[0] == 1 else 'Negativo'
    else:
        return 'Idioma no soportado'

# Interfaz de usuario con Streamlit
st.title('Clasificación de Comentarios de Videojuegos')
st.write('Introduce un comentario y el modelo clasificará si es positivo o negativo.')

comentario = st.text_area("Comentario")

if st.button('Clasificar'):
    if comentario:
        resultado = predict_comment(comentario)
        st.write(f"El comentario: '{comentario}' es {resultado}")
    else:
        st.write("Por favor, introduce un comentario.")

# Para ejecutar la aplicación, usa el siguiente comando en la terminal:
# streamlit run nombre_del_archivo.py
