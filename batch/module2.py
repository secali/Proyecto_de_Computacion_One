import sys
import os
import batch.functions

import pandas as pd


def batchTwo ():
    # obtener ruta fichero a cargar
    print("\nCargando fichero...")
    file = batch.functions.obtener_ruta()

    # creamos dataframe con datos fichero
    dfDataSet=pd.read_csv(file, delimiter='\t')
    print (dfDataSet)

    # imprimir primera tabla pedida

    # Calcular el número total de instancias
    n_total = len(dfDataSet)
    # Filtrar instancias humanas y generadas
    dfHumano = dfDataSet[dfDataSet['Type'] == 'h']
    dfGeneradas = dfDataSet[dfDataSet['Type'] == 'g']
    # Número de instancias humanas y generadas
    n_humano = len(dfHumano)
    n_generadas = len(dfGeneradas)

    # Longitud media de caracteres para instancias humanas y generadas
    long_media_humano = dfHumano[
        'Text'].str.len().mean()
    long_media_generadas = dfGeneradas[
        'Text'].str.len().mean()

    print(f"Número total de instancias: {n_total}")
    print(f"Número de instancias humanas: {n_humano}")
    print(f"Número de instancias generadas: {n_generadas}")
    print(f"Longitud media de instancias humanas: {long_media_humano:.2f}")
    print(f"Longitud media de instancias generadas: {long_media_generadas:.2f}")
    '''# Mostrar los resultados
    print(f"Número total de instancias: {n_total}")
    print(f"Número de instancias humanas: {n_humano}")
    print(f"Número de instancias generadas: {n_generadas}")
    print(f"Longitud media de instancias humanas: {long_media_humano:.2f}")
    print(f"Longitud media de instancias generadas: {long_media_generadas:.2f}")'''

    sys.exit()


