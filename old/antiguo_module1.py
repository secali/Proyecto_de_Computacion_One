# import required libraries

import json
import gdown
import pandas as pd
import batch.functions


# batch 1 - lo usamos para hacer el crawling y el scraping
def batchOne():
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

        batch.functions.guardar_dataset(pd.concat([dfHuman, dfGenerated]), 'DataFrame.tsv')

    # simulamos el salto al modulo 2, que continuaria con las operaciones requeridas.
    batch.module2.batchTwo()
