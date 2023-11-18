import sys
import batch.functions
from tabulate import tabulate
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

    # Creamos una lista con los datos
    data =[
        ["Número total de instancias", n_total],
        ["Número de instancias humanas", n_humano],
        ["Número de instancias generadas", n_generadas],
        ["Longitud media de instancias humanas", f"{long_media_humano:.2f}"],
        ["Longitud media de instancias generadas", f"{long_media_generadas:.2f}"]
        ]
    # imprimimos los datos en forma de tabla tabulada
    print(tabulate(data, headers=["Campo", "Valor"], tablefmt="grid"))






    sys.exit()


