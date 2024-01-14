# import required libraries

import batch.functions
import batch.module3
from tabulate import tabulate
import pandas as pd
import numpy as np


# batch 2 - modulo que usamos para cargar fichero, hacer estadísticas y balancear los datos
def batchTwo():
    print("\n############ Ejecutando Batch 2: carga fichero, estdística y manejo de datos #############")
    # obtener ruta fichero a cargar
    print("\nCargando fichero...")
    file = batch.functions.obtener_ruta_guardado('SaveDF', 'DataFrame.tsv')

    # creamos dataframe con datos fichero
    dfDataSet = pd.read_csv(file, delimiter='\t')
    print(dfDataSet)

    # calcular el número total de instancias
    n_total = len(dfDataSet)

    # dividir instancias humanas y generadas
    dfHuman = dfDataSet[dfDataSet['label'] == '0']
    dfIA = dfDataSet[dfDataSet['label'] == '1']

    # número de instancias humanas y generadas
    n_humano = len(dfHuman)
    n_generadas = len(dfIA)

    # longitud media de caracteres para instancias humanas y generadas
    long_media_humano = dfHuman['text'].str.len().mean()
    long_media_generadas = dfIA['text'].str.len().mean()

    # creamos una lista de listas con los datos
    data = [
        ["Número total de instancias", n_total],
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
        indices_to_remove = np.random.choice(dfIA.index, (n_generadas - n_humano), replace=False)
        # ajustamos el dataframe al tamaño adecuado
        dfIA = dfIA.drop(indices_to_remove)

    # construimos el dataset final
    dfDataSet_final = pd.concat([dfHuman, dfIA], ignore_index=True)

    # guardamos el DataSet final
    batch.functions.guardar_dataset(dfDataSet_final, 'DataSetFinal.tsv')

    batch.module3.batchThree()
