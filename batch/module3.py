# import required libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import batch.module4
import batch.functions
from tabulate import tabulate


def batchThree():
    print("\n############ Ejecutando Batch 3: Clasificador #############")
    max_instances_per_class = 100
    max_features = 1000  # maximum number of features extracted for our instances
    random_seed = 777  # set random seed for reproducibility
    id2label = {0: "h", 1: "g"}


    print("\nCargando fichero...")
    file = batch.functions.obtener_ruta_guardado('SaveDF','DataSetFinal.tsv')

    # creamos dataframe con datos fichero
    dfDataSet = pd.read_csv(file, delimiter='\t')
    print(dfDataSet)

    # Separar las características (X) del objetivo (y)
    X = dfDataSet['Text']
    y = dfDataSet['Type']

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

    # Crear DataFrames para los conjuntos de entrenamiento y prueba
    train_df = pd.DataFrame({'Text': X_train, 'Type': y_train})
    test_df = pd.DataFrame({'Text': X_test, 'Type': y_test})

    # print(test_df)
    # downsample training data to train faster
    train_df = train_df.groupby("Type").sample(n=max_instances_per_class, random_state=random_seed)
    # vectorize data: extract features from our data (from text to numeric vectors)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))
    X_train = vectorizer.fit_transform(train_df["Text"])
    X_test = vectorizer.transform(test_df["Text"])

    # vectorize labels : from text to numeric vectors
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Type"])
    y_test = le.transform(test_df["Type"])

    # print (y_test)

    # Calcular mejor algoritmo
    best_score = float('-inf')
    best_model = None

    # recorremos los algoritmos de clasificacion
    for name, ClassifierClass in all_estimators(type_filter='classifier'):
        if issubclass(ClassifierClass, ClassifierMixin) and hasattr(ClassifierClass, 'fit'):
            try:
                regressor = ClassifierClass()
                print (regressor)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                # print(y_pred)
                score = f1_score(y_test, y_pred, average="macro")
                if score > best_score:
                    best_score = score
                    best_model = regressor
                print(f"Model: {name} Macro F1: {score:.3f}")
            except Exception as e:
                print(f"Error en el modelo {name}: {e}")
                continue
    # imprimimos los resultados
    print(f"\nBest Model: {best_model.__class__.__name__}")
    print(f"Macro F1 on Test Data: {best_score}")

    try:
        print (best_model)
        best_model.fit(X_train, y_train)
        print(f"Modelo: {best_model.__class__.__name__} entrenado.\nGuardando clasificador...")
    except Exception as e:
        print(f"Error al entrenar el mejor modelo: {e}")

    batch.functions.guardar_clf_vct('clf', best_model)
    batch.functions.guardar_clf_vct('vct',vectorizer)

    batch.module4.batchFour()
