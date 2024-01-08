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
import batch.antiguo_module2
import batch.functions
import http.client


# batch 1 - lo usamos para hacer el crawling y el scraping
def scriptExecution():
    print("\n############Ejecutando Batch 1: Crawling y Scraping#############\n")
    # TODO: CONECTAR CON GOOGLE DRIVE Y OBTENER LOS DATOS CON GDOWN, LUEGO PARSEAMOS A JSON
    url_dev_subtaskA_mono = 'https://drive.google.com/file/d/1e_G-9a66AryHxBOwGWhriePYCCa4_29e/view'
    url_train_subtaskA_mono ='https://drive.google.com/file/d/1HeCgnLuDoUHhP-2OsTSSC3FXRLVoI6OG/view'
    output_dev_subtaskA_mono = 'downloaded_dev_data_taskA.jsonl'
    output_train_subtaskA_mono = 'downloaded_train_data_taskA.jsonl'
    gdown.download(url_dev_subtaskA_mono, output_dev_subtaskA_mono, fuzzy=True)
    gdown.download(url_train_subtaskA_mono, output_train_subtaskA_mono, fuzzy=True)

    dfHuman = pd.DataFrame()
    dfGenerated = pd.DataFrame()

    with open(output_dev_subtaskA_mono, 'r') as f:
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
        ["Número total de instancias", n_humano+n_generadas],
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
