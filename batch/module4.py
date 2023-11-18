from joblib import load
from batch import functions


def batchFour():
    print("\n############ Ejecutando Batch 4: Carga clasificador y realizar predicciones #############")

    file,ruta =functions.obtener_ruta_guardado_clf()
    print (ruta+file)

    # Cargar el clasificador desde el archivo
    loaded_classifier = load(ruta+file)
    '''
    # Suponiendo que tienes nuevos datos 'X_new' para hacer predicciones
    # Reemplaza 'X_new' con tus propios datos
    # Debes asegurarte de que 'X_new' tenga la misma estructura que los datos utilizados para entrenar el clasificador

    # Realizar predicción con los nuevos datos
    predictions = loaded_classifier.predict("buenos dias, soy humano")
    print(predictions)

    # Las predicciones se encuentran ahora en la variable 'predictions'
    # Puedes usar estas predicciones según sea necesario
    '''