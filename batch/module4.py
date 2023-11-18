import sys

from joblib import load
from batch import functions
from sklearn.feature_extraction.text import TfidfVectorizer


def batchFour():
    print("\n############ Ejecutando Batch 4: Carga clasificador y realizar predicciones #############")

    file,ruta =functions.obtener_ruta_guardado_clf()

        # Cargar el clasificador desde el archivo
    loaded_classifier = load(ruta+file)
    print("Clasificador "+file+ " cargado")
    # Crear un vectorizador con las mismas configuraciones que usamos en entrenamiento
    max_features = 2000
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))
    valores_aceptados = ['S', 'N']
    entrada = 'a'
    while entrada != valores_aceptados:
        entrada = input("\n¿Desea usted comprobar un texto? S/N:\n")
        if entrada in ['S', 's']:
            texto=input("Ingresa el texto:")
            texto_vectorizado = vectorizer.transform([texto])
            # Realizar la predicción con el modelo cargado
            prediction = loaded_classifier.predict(texto_vectorizado)
            # Imprimir la predicción
            print(f"La predicción para la cadena de texto es: {prediction}")
        elif entrada in ['N', 'n']:
            print ("Aplicacion finalizada.\nCerrandose aplicacion")
            sys.exit()
        else:
            print("Seleccione una opción válida")
