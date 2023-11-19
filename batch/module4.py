import sys

import pandas as pd
from joblib import load
from batch import functions
from sklearn.feature_extraction.text import TfidfVectorizer


def batchFour():
    print("\n############ Ejecutando Batch 4: Carga clasificador y realizar predicciones #############")

    # Cargar el clasificador desde el archivo
    loaded_classifier = load(functions.obtener_ruta_guardado('SaveCLF', 'clf.joblib'))
    print("Clasificador cargado")
    # Cargar el vectorizador desde un archivo
    loaded_vectorizador = load(functions.obtener_ruta_guardado('SaveVCT', 'vct.joblib'))
    print("Vectorizador cargado")

    valores_aceptados = ['S', 'N']
    entrada = 'a'
    while entrada != valores_aceptados:
        entrada = input("\n¿Desea usted comprobar un texto? S/N:\n")
        if entrada in ['S', 's']:
            texto = input("Ingresa el texto:")
            data = {'Text': [texto],
                    'Type': ['f']}
            dfTexto = pd.DataFrame(data, index=['Fila_1'])
            texto_vectorizado = loaded_vectorizador.transform(dfTexto["Text"])
            y_pred = loaded_classifier.predict(texto_vectorizado)
            if y_pred == 0:
                print('El texto introducido ha sido escrito por un humano')
            else:
                print('El texto inntroducio ha sido generado por IA')
        elif entrada in ['N', 'n']:
            print("Aplicacion finalizada.\nCerrandose aplicacion")
            sys.exit()
        else:
            print("Seleccione una opción válida")
