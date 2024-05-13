import pickle
import os

# Cargar el modelo
with open('modelo.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar el vectorizador
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def cargar_correos(carpeta):
    X = []  # Lista para almacenar el texto de los correos electrónicos

    # Leer correos electrónicos
    archivos = os.listdir(carpeta)
    for archivo in archivos:
        ruta_completa = os.path.join(carpeta, archivo)
        if os.path.isfile(ruta_completa):
            with open(ruta_completa, "r", encoding="latin-1") as f:
                contenido = f.read()
                X.append(contenido)

    return X

correos = cargar_correos("test")

# Transformar los correos utilizando el vectorizador cargado
correos_transformados = vectorizer.transform(correos)

# Usar el modelo cargado para hacer predicciones en los correos
predicciones = model.predict(correos_transformados)

# Imprimir las predicciones
for i in range(len(correos)):
    print("Correo #", i+1, "es", "spam" if predicciones[i] == 1 else "no spam")