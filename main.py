import os
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QFileDialog
from mw import Ui_MainWindow

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.X = []
        self.y = []
        self.model = None
        self.vectorizer = None
        self.ui.pushButton_Entradas.clicked.connect(self.entradas)
        self.ui.pushButton_Salida.clicked.connect(self.salidas)
        self.ui.pushButton_reset.clicked.connect(self.reset)
        self.ui.pushButton_Entrenar.clicked.connect(self.entrenar)
        
        # importar modelo
        self.ui.pushButton_importar.clicked.connect(self.importar)
        self.ui.pushButton_entradas_importar.clicked.connect(self.entradas_importar)
        self.ui.pushButton_clasificar.clicked.connect(self.clasificar)

    def clasificar(self):
        correos_transformados = self.vectorizer.transform(self.X)

        # Usar el modelo cargado para hacer predicciones en los correos
        predicciones = self.model.predict(correos_transformados)

        # Imprimir las predicciones
        spam = 0
        nospam = 0
        for i in range(len(self.X)):
            if predicciones[i] == 1 :
                texto = "Correo #" + str((i+1)) + "es spam"
                spam += 1
                self.ui.textBrowser_spam.append(str(texto))
                
            else:
                texto = "Correo #" + str((i+1)) + "No es spam"
                nospam +=1
                self.ui.textBrowser_nospam.append(str(texto))
        
        porcentaje_spam = spam * 100 / len(self.X) 
        porcentaje_nospam = nospam * 100 / len(self.X) 
        
        self.ui.lineEdit_porc_no_spam.setText(str(porcentaje_nospam))
        self.ui.lineEdit_porc_spam.setText(str(porcentaje_spam))
        self.ui.lineEdit_num_correos_spam.setText(str(spam))
        self.ui.lineEdit_num_correros_no_spam.setText(str(nospam))
    def entradas_importar(self):
        ruta_carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta")
        nombre_carpeta = os.path.basename(ruta_carpeta)        
        archivos = os.listdir(nombre_carpeta)
        for archivo in archivos:
            ruta_completa = os.path.join(nombre_carpeta, archivo)
            if os.path.isfile(ruta_completa):
                with open(ruta_completa, "r", encoding="latin-1") as f:
                    contenido = f.read()
                    self.X.append(contenido)
    def importar(self):
        with open('modelo.pkl', 'rb') as f:
            self.model = pickle.load(f)

        # Cargar el vectorizador
        with open('vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

        
    def entrenar(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)  # Limitamos a las 1000 palabras más frecuentes
        X_tfidf = self.vectorizer.fit_transform(self.X)
        # Crear y entrenar el modelo MLP
        self.model = MLPClassifier(hidden_layer_sizes=(6), max_iter=1000, random_state=42)
        self.model.fit(X_tfidf, self.y)

        # Predecir en todos los datos
        y_pred = self.model.predict(X_tfidf)

        # Calcular la matriz de confusión
        conf_matrix = confusion_matrix(self.y, y_pred)
        print("Matriz de Confusión:")
        print(conf_matrix)

        # Calcular la precisión del modelo
        accuracy = accuracy_score(self.y, y_pred)
        print("Precisión del modelo:", accuracy)
        # # Guardar el modelo
        with open('modelo.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        # Guardar el vectorizador
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)


    def reset(self):
        self.X.clear()
        self.y.clear()
        self.ui.textBrowser_nospam.clear()
        self.ui.textBrowser_spam.clear()

    def salidas(self):
        ruta_carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta")
        nombre_carpeta = os.path.basename(ruta_carpeta)        
        archivos_spam = os.listdir(nombre_carpeta)
        for archivo in archivos_spam:
            ruta_completa = os.path.join(nombre_carpeta, archivo)
            if os.path.isfile(ruta_completa):
                with open(ruta_completa, "r", encoding="latin-1") as f:
                    contenido = f.read()
                    self.X.append(contenido)
                    self.y.append(1)  # Etiqueta 1 para spam

    def entradas(self):
        ruta_carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta")
        nombre_carpeta = os.path.basename(ruta_carpeta)
        archivos_no_spam = os.listdir(nombre_carpeta)
        for archivo in archivos_no_spam:
            ruta_completa = os.path.join(nombre_carpeta, archivo)
            if os.path.isfile(ruta_completa):
                with open(ruta_completa, "r", encoding="latin-1") as f:
                    contenido = f.read()
                    self.X.append(contenido)
                    self.y.append(0)  # Etiqueta 0 para no spam
        
# Función para cargar los datos y preprocesarlos

app = QApplication(sys.argv)
ventana = Window()
ventana.show()
sys.exit(app.exec_())


# def cargar_datos():
#     X = []  # Lista para almacenar el texto de los correos electrónicos
#     y = []  # Lista para almacenar las etiquetas (0 para no spam, 1 para spam)

#     # Leer correos electrónicos que no son spam
#     carpeta_no_spam = "easy_ham"
#     archivos_no_spam = os.listdir(carpeta_no_spam)
#     for archivo in archivos_no_spam:
#         ruta_completa = os.path.join(carpeta_no_spam, archivo)
#         if os.path.isfile(ruta_completa):
#             with open(ruta_completa, "r", encoding="latin-1") as f:
#                 contenido = f.read()
#                 X.append(contenido)
#                 y.append(0)  # Etiqueta 0 para no spam

#     # Leer correos electrónicos de spam
#     carpeta_spam = "spam"
#     archivos_spam = os.listdir(carpeta_spam)
#     for archivo in archivos_spam:
#         ruta_completa = os.path.join(carpeta_spam, archivo)
#         if os.path.isfile(ruta_completa):
#             with open(ruta_completa, "r", encoding="latin-1") as f:
#                 contenido = f.read()
#                 X.append(contenido)
#                 y.append(1)  # Etiqueta 1 para spam
#     return X, y

# # Cargar los datos
# X, y = cargar_datos()
# # Preprocesamiento de texto usando TF-IDF
# vectorizer = TfidfVectorizer(max_features=1000)  # Limitamos a las 1000 palabras más frecuentes
# X_tfidf = vectorizer.fit_transform(X)
# # Crear y entrenar el modelo MLP
# model = MLPClassifier(hidden_layer_sizes=(6), max_iter=1000, random_state=42)
# model.fit(X_tfidf, y)

# # Predecir en todos los datos
# y_pred = model.predict(X_tfidf)

# # Calcular la matriz de confusión
# conf_matrix = confusion_matrix(y, y_pred)
# print("Matriz de Confusión:")
# print(conf_matrix)

# # Calcular la precisión del modelo
# accuracy = accuracy_score(y, y_pred)
# print("Precisión del modelo:", accuracy)

# # Guardar el modelo
# with open('modelo.pkl', 'wb') as f:
#     pickle.dump(model, f)

# # Guardar el vectorizador
# with open('vectorizer.pkl', 'wb') as f:
#     pickle.dump(vectorizer, f)
