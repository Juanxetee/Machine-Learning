import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el modelo y el vectorizador
with open('Modelo/best_modelfinal_rf.pkl', 'rb') as file:
    best_model_rf = pickle.load(file)
    
with open('Modelo/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Función para predecir el comentario
def predecir_comentario():
    comentario = entrada_comentario.get("1.0", tk.END).strip()
    if comentario:
        comentario_vectorizado = vectorizer.transform([comentario])
        prediccion = best_model_rf.predict(comentario_vectorizado)
        resultado = "Positivo" if prediccion[0] == 1 else "Negativo"
        messagebox.showinfo("Resultado de la Predicción", f"El comentario es {resultado}")
        entrada_comentario.delete("1.0", tk.END)  # Limpiar el campo de entrada después de la predicción
    else:
        messagebox.showwarning("Entrada Vacía", "Por favor ingrese un comentario.")

# Crear la interfaz gráfica
ventana = tk.Tk()
ventana.title("Clasificación de Comentarios")
ventana.geometry("800x600")  # Ajustar el tamaño de la ventana
ventana.configure(bg='black')  # Fondo negro

# Etiqueta y campo de entrada para el comentario
etiqueta_comentario = tk.Label(ventana, text="Ingrese un comentario:", font=("Helvetica", 14), fg="white", bg="black")
etiqueta_comentario.pack(pady=10)

entrada_comentario = tk.Text(ventana, height=10, width=70, font=("Helvetica", 12), bg="black", fg="white", insertbackground="white")
entrada_comentario.pack(pady=10)

# Botón para realizar la predicción
boton_predecir = tk.Button(ventana, text="Predecir", font=("Helvetica", 12, "bold"), bg="#ff6347", fg="white", command=predecir_comentario)
boton_predecir.pack(pady=10)

# Ejecutar la interfaz gráfica
ventana.mainloop()

