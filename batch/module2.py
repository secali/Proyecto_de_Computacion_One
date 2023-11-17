import sys
import os
import batch.functions

import pandas as pd


def batchTwo ():
    # obtener ruta fichero a cargar
    file = batch.functions.obtener_ruta()

    # cargamos dataframe con datos fichero
    dfDataSet=pd.read_csv(file, delimiter='\t')
    print (dfDataSet)

    sys.exit()


