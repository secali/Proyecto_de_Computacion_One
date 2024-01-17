import sys

import pandas as pd
from joblib import load
from batch import functions


# batch 4 - modulo que usamos para cargar clasificador y vectorizador y realizar predicciones
# el vectorizador se utiliza para convertir el texto en una representación numérica antes de alimentarlo al modelo de clasificación
def batchFour():
    print("\n############ Ejecutando Batch 4: Carga clasificador y realizar predicciones #############")

    # cargamos el clasificador desde el archivo
    loaded_classifier = load(functions.obtener_ruta_guardado('SaveCLF', 'clf.joblib'))
    print("Clasificador cargado")
    # cargar el vectorizador desde un archivo
    loaded_vectorizador = load(functions.obtener_ruta_guardado('SaveVCT', 'vct.joblib'))
    print("Vectorizador cargado")

    # bucle para indicar si queremos hacer alguna predicción
    valores_aceptados = ['S', 'N']
    entrada = 'a'
    while entrada != valores_aceptados:
        entrada = input("\n¿Desea usted comprobar un texto? S/N:\n")
        if entrada in ['S', 's']:
            texto = input("Ingresa el texto:")
            data = {'Text': [texto],
                    'Label': ['f']}
            # preparamos texto introducido
            dfTexto = pd.DataFrame(data, index=['Fila_1'])
            texto_vectorizado = loaded_vectorizador.transform(dfTexto["Text"])
            # lanzamos la prediccion
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
