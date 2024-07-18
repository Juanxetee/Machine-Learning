import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

with open('Modelo/best_modelfinal_rf.pkl', 'rb') as file:
    best_model_rf = pickle.load(file)
    
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Función para predecir el comentario
def predecir_comentario():
    comentario = entrada_comentario.get("1.0", tk.END).strip()
    if comentario:
        comentario_vectorizado = vectorizer.transform([comentario])
        prediccion = best_model_rf.predict(comentario_vectorizado)
        resultado = "Positivo" if prediccion[0] == 1 else "Negativo"
        messagebox.showinfo("Resultado de la Predicción", f"El comentario es {resultado}")
    else:
        messagebox.showwarning("Entrada Vacía", "Por favor ingrese un comentario.")

# Crear la interfaz gráfica
ventana = tk.Tk()
ventana.title("Clasificación de Comentarios")

# Etiqueta y campo de entrada para el comentario
etiqueta_comentario = tk.Label(ventana, text="Ingrese un comentario:")
etiqueta_comentario.pack(pady=5)

entrada_comentario = tk.Text(ventana, height=10, width=50)
entrada_comentario.pack(pady=5)

# Botón para realizar la predicción
boton_predecir = tk.Button(ventana, text="Predecir", command=predecir_comentario)
boton_predecir.pack(pady=5)

# Ejecutar la interfaz gráfica
ventana.mainloop()
