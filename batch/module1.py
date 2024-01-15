# import required libraries

import requests
import json
import gdown
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import numpy as np
from langdetect import detect
import pandas as pd
import batch.module2
import batch.functions
import http.client
import urllib.request

# batch 1 - lo usamos para hacer el crawling y el scraping
def batchOne():
    print("\n############ Ejecutando Batch 1: Descarga de ficheros #############\n")
    # TODO: CONECTAR CON GOOGLE DRIVE Y OBTENER LOS DATOS CON GDOWN, LUEGO PARSEAMOS A JSON

    archivos = [
        ('https://drive.google.com/file/d/1e_G-9a66AryHxBOwGWhriePYCCa4_29e/view', 'subtaskA_dev_monolingual.jsonl'),
        ('https://drive.google.com/file/d/1HeCgnLuDoUHhP-2OsTSSC3FXRLVoI6OG/view', 'subtaskA_train_monolingual.jsonl'),
        ('https://drive.google.com/file/d/1oh9c-d0fo3NtETNySmCNLUc6H1j4dSWE/view', 'subtaskB_dev.jsonl'),
        ('https://drive.google.com/file/d/1k5LMwmYF7PF-BzYQNE2ULBae79nbM268/view', 'subtaskB_train.jsonl')
    ]
    # descargamos los archivos
    batch.functions.descarga_archivos(archivos)

    batch.module2.batchTwo()

'''
    url_dev_subtaskA = 'https://drive.google.com/file/d/1e_G-9a66AryHxBOwGWhriePYCCa4_29e/view'
    url_train_subtaskA = 'https://drive.google.com/file/d/1HeCgnLuDoUHhP-2OsTSSC3FXRLVoI6OG/view'
    url_dev_subtaskB = 'https://drive.google.com/file/d/1oh9c-d0fo3NtETNySmCNLUc6H1j4dSWE/view'
    url_train_subtaskB = 'https://drive.google.com/file/d/1k5LMwmYF7PF-BzYQNE2ULBae79nbM268/view'

    output_dev_subtaskA = batch.functions.obtener_ruta_guardado('Descargas', 'downloaded_dev_data_taskA.jsonl')
    output_train_subtaskA = batch.functions.obtener_ruta_guardado('Descargas', 'downloaded_train_data_taskA.jsonl')
    output_dev_subtaskB = batch.functions.obtener_ruta_guardado('Descargas', 'downloaded_dev_data_taskB.jsonl')
    output_train_subtaskB = batch.functions.obtener_ruta_guardado('Descargas', 'downloaded_train_data_taskB.jsonl')
    gdown.download(url_dev_subtaskA, output_dev_subtaskA, fuzzy=True)
    gdown.download(url_train_subtaskA, output_train_subtaskA, fuzzy=True)
    gdown.download(url_dev_subtaskB, output_dev_subtaskB, fuzzy=True)
    gdown.download(url_train_subtaskB, output_train_subtaskB, fuzzy=True)
    


    dfHuman = pd.DataFrame()
    dfGenerated = pd.DataFrame()

    with open(output_dev_subtaskA, 'r') as f:
        for line in f:
            # Decodificar cada línea como un objeto JSON
            # print(line)
            decoded_data = json.loads(line)

            # Convertir el objeto JSON en una serie de pandas
            pandas_data = pd.Series(decoded_data)

            if decoded_data['label'] == 0:
                # print('Is human')
                dfHuman = dfHuman._append(pandas_data, ignore_index=True)
            elif decoded_data['label'] == 1:
                # print('Is generated')
                dfGenerated = dfGenerated._append(pandas_data, ignore_index=True)

        print(dfHuman)
        print(dfGenerated)

    print("\n############ Ejecutando Batch 2: carga fichero, estdística y manejo de datos #############")
    # Batch 2
    # número de instancias humanas y generadas
    n_humano = len(dfHuman)
    n_generadas = len(dfGenerated)

    # longitud media de caracteres para instancias humanas y generadas
    long_media_humano = dfHuman['text'].str.len().mean()
    long_media_generadas = dfGenerated['text'].str.len().mean()

    # creamos una lista de listas con los datos
    data = [
        ["Número total de instancias", n_humano + n_generadas],
        ["Número de instancias humanas", n_humano],
        ["Número de instancias generadas", n_generadas],
        ["Longitud media de instancias humanas", f"{long_media_humano:.2f}"],
        ["Longitud media de instancias generadas", f"{long_media_generadas:.2f}"]
    ]
    # imprimimos los datos en forma de tabla tabulada
    print(tabulate(data, headers=["Campo", "Valor"], tablefmt="grid"))

    # balanceamos el numero de resultados
    if n_humano > n_generadas:
        # elegimos de forma random los indices a borrar
        indices_to_remove = np.random.choice(dfHuman.index, (n_humano - n_generadas), replace=False)
        # ajustamos el dataframe al tamaño adecuado
        dfHuman = dfHuman.drop(indices_to_remove)
    if n_humano < n_generadas:
        # elegimos de forma random los indices a borrar
        indices_to_remove = np.random.choice(dfGenerated.index, (n_generadas - n_humano), replace=False)
        # ajustamos el dataframe al tamaño adecuado
        dfGenerated = dfGenerated.drop(indices_to_remove)

    # construimos el dataset final
    dfDataSet_final = pd.concat([dfHuman, dfGenerated], ignore_index=True)

    batch.functions.guardar_dataset(dfDataSet_final, 'DataSetFinal.tsv')

    batch.module3.batchThree()
    '''