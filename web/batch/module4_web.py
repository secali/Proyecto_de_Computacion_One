import sys

import pandas as pd
from joblib import load
from batch import functions


# batch 4 - modulo que usamos para cargar clasificador y vectorizador y realizar predicciones
# el vectorizador se utiliza para convertir el texto en una representación numérica antes de alimentarlo al modelo de clasificación
def batchFour(modelo, texto):
    print("\n############ Ejecutando Batch 4: Carga clasificador y realizar predicciones #############")

    if modelo == 'subtareaA':
        # cargamos el clasificador desde el archivo
        loaded_classifier_A = load(functions.obtener_ruta_guardado('SaveCLF', 'clf_A.joblib'))
        print("Clasificador A cargado")
        # cargar el vectorizador desde un archivo
        loaded_vectorizador_A = load(functions.obtener_ruta_guardado('SaveVCT', 'vct_A.joblib'))
        print("Vectorizador A cargado")

        test_df = pd.DataFrame()
        test_df ['texto'] = [texto]
        texto_vectorizado = loaded_vectorizador_A.transform(test_df ['texto'])
        # lanzamos la prediccion
        y_pred = loaded_classifier_A.predict(texto_vectorizado)
        if y_pred == 0:
           return 'El texto introducido ha sido escrito por un humano'
        else:
            return 'El texto introducio ha sido generado por IA'

    elif modelo == 'subtareaB':

        # cargamos el clasificador desde el archivo
        loaded_classifier_B = load(functions.obtener_ruta_guardado('SaveCLF', 'clf_B.joblib'))
        print("Clasificador B cargado")
        # cargar el vectorizador desde un archivo
        loaded_vectorizador_B = load(functions.obtener_ruta_guardado('SaveVCT', 'vct_B.joblib'))
        print("Vectorizador B cargado")


        texto_vectorizado = loaded_vectorizador_B.transform(texto)
        # lanzamos la prediccion
        y_pred = loaded_classifier_B.predict(texto_vectorizado)
        if y_pred == 0:
            print('El texto introducido ha sido escrito por un humano')
        elif y_pred == 1:
            print('El texto introducio ha sido generado por ChatGPT')
        elif y_pred == 2:
            print('El texto introducio ha sido generado por Cohere')
        elif y_pred == 3:
            print('El texto introducio ha sido generado por Davinci')
        elif y_pred == 4:
            print('El texto introducio ha sido generado por Bloomz')
        else:
            print('El texto introducio ha sido generado por Dolly')
