import sys
import batch.functions
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def batchTwo():
    # obtener ruta fichero a cargar
    print("\nCargando fichero...")
    file = batch.functions.obtener_ruta()

    # creamos dataframe con datos fichero
    dfDataSet = pd.read_csv(file, delimiter='\t')
    print(dfDataSet)

    # imprimir primera tabla pedida

    # Calcular el número total de instancias
    n_total = len(dfDataSet)
    # Filtrar instancias humanas y generadas
    dfHuman = dfDataSet[dfDataSet['Type'] == 'h']
    dfIA = dfDataSet[dfDataSet['Type'] == 'g']
    # Número de instancias humanas y generadas
    n_humano = len(dfHuman)
    n_generadas = len(dfIA)

    # Longitud media de caracteres para instancias humanas y generadas
    long_media_humano = dfHuman[
        'Text'].str.len().mean()
    long_media_generadas = dfIA[
        'Text'].str.len().mean()

    # Creamos una lista de listas con los datos
    data = [
        ["Número total de instancias", n_total],
        ["Número de instancias humanas", n_humano],
        ["Número de instancias generadas", n_generadas],
        ["Longitud media de instancias humanas", f"{long_media_humano:.2f}"],
        ["Longitud media de instancias generadas", f"{long_media_generadas:.2f}"]
    ]
    # Imprimimos los datos en forma de tabla tabulada
    print(tabulate(data, headers=["Campo", "Valor"], tablefmt="grid"))

    # Balanceamos el numero de resultados
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

    # Separar las características (X) del objetivo (y)
    X = dfDataSet_final['Text']  # Características, en este caso, la columna 'Text'
    y = dfDataSet_final['Type']  # Objetivo, en este caso, la columna 'Type'

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Número total de instancias en el dataset original
    n_total = len(dfDataSet)
    # Número de instancias en el conjunto de entrenamiento
    n_train = len(X_train)
    # Número de instancias en el conjunto de prueba
    n_test = len(X_test)
    # Número de instancias humanas en el conjunto de entrenamiento
    n_human_train = sum(y_train == 'h')
    # Número de instancias generadas en el conjunto de entrenamiento
    n_generated_train = sum(y_train == 'g')
    # Número de instancias humanas en el conjunto de prueba
    n_human_test = sum(y_test == 'h')
    # Número de instancias generadas en el conjunto de prueba
    n_generated_test = sum(y_test == 'g')

    # Creamos una lista de listas con los datos
    data = [
        ["Número de instancias del training", n_train],
        ["Número de instancias del test", n_test],
        ["Número de instancias humanas en el training", n_human_train],
        ["Número de instancias generadas en el training", n_generated_train],
        ["Número de instancias humanas en el test", n_human_test],
        ["Número de instancias generadas en el test", n_generated_test]
    ]

    # Imprimimos los datos en forma de tabla tabulada
    print(tabulate(data, headers=["Descripción", "Valor"], tablefmt="grid"))

    # print(y_train + " - " + X_train + "\n")
    # print(y_test + " - " + X_test + "\n")




    sys.exit()
