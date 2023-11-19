import sys

import pandas as pd
from joblib import load
from batch import functions
from sklearn.feature_extraction.text import TfidfVectorizer


def batchFour():
    print("\n############ Ejecutando Batch 4: Carga clasificador y realizar predicciones #############")

    file_clf,ruta_clf =functions.obtener_ruta_guardado_clf()
    file_vct, ruta_vct = functions.obtener_ruta_guardado_vct()

    # Cargar el clasificador desde el archivo
    loaded_classifier = load(ruta_clf+file_clf)
    print("Clasificador "+file_clf+ " cargado")
    # Cargar el vectorizador desde un archivo
    loaded_vectorizador = load(ruta_vct+file_vct)
    print("Vectorizador " + file_vct + " cargado")

    valores_aceptados = ['S', 'N']
    entrada = 'a'
    while entrada != valores_aceptados:
        entrada = input("\n¿Desea usted comprobar un texto? S/N:\n")
        if entrada in ['S', 's']:
            texto=input("Ingresa el texto:")

            df = pd.DataFrame({'Text':pd.Series(texto)})
            print (df)
            print (texto)
            X_test = loaded_vectorizador.transform(df["Text"])
            print (X_test.shape)
            # Realizar la predicción con el modelo cargado
            prediction = loaded_classifier.predict(X_test)
            # Imprimir la predicción
            print(f"La predicción para la cadena de texto es: {prediction}")
        elif entrada in ['N', 'n']:
            print ("Aplicacion finalizada.\nCerrandose aplicacion")
            sys.exit()
        else:
            print("Seleccione una opción válida")
