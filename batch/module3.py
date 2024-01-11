# import required libraries
import sys

import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

import batch.functions
from tabulate import tabulate
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.datasets import load_iris
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
#from gensim.models import Word2Vec
#from gensim.models import Doc2Vec, TaggedDocument

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# batch 3 - modulo que usamos para crear los test, elegir modelo, entrenar y guardar clasificador y vectorizador
def batchThree():
    print("\n############ Ejecutando Batch 3: Clasificador #############")
    # creamos y asignamos valor a las variables
    max_instances_per_class = 500 #100  # numero de instancias por clase
    max_features = 3000  # maximum number of features extracted for our instances
    random_seed = 777  # set random seed for reproducibility

    # obtener ficheros a cargar
    print("\nCargando ficheros...")
    fileATrain = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTrain_A.tsv')
    fileATest = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTest_A.tsv')
    # fileFase1 = batch.functions.obtener_ruta_guardado('SaveDF', 'DataSetFinal.tsv')
    fileFase1 = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTest_fase01.tsv')

    # creamos dataframe con datos de los ficheros
    print("\nCreando DataFrames...")
    df_train_A = pd.read_csv(fileATrain, delimiter='\t')
    df_test_A = pd.read_csv(fileATest, delimiter='\t')
    df_fase_1 = pd.read_csv(fileFase1, delimiter='\t')

    # Imrpimimos estadistica
    batch.functions.imprime_estadistica_subtarea_A(df_train_A, df_test_A, df_fase_1)

    # Balanceando fichero de train
    print("Balanceamos los ficheros")
    df_train_A = batch.functions.balacearDF(df_train_A)
    df_test_A = batch.functions.balacearDF(df_test_A)
    # Imrpimimos estadistica
    print("\nEstadistica con fichero balanceado")
    batch.functions.imprime_estadistica_subtarea_A(df_train_A, df_test_A, df_fase_1)

    print("\nPreparando datos para hacer entrenamiento y test")
    '''# Crear DataFrames para los conjuntos de entrenamiento y prueba
    train_df = df_train_A  # pd.DataFrame({'text': X_train, 'label': y_train})  # 'Type': y_train})
    test_df = df_test_A  # pd.DataFrame({'text': X_test, 'label': y_test})  # X_test, 'Type': y_test})
    # batch.functions.guardar_dataset(test_df, 'test_df_borrar.tsv')
    test_df_f01 = df_fase_1  # pd.DataFrame({'text': X_test_f01, 'label': y_test_f01})  # X_test, 'Type': y_test})'''

    # Separar las características (X) del objetivo (y)
    X = df_train_A['text']
    y = df_train_A['label']

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    print("Creando ficheros de entranamiento y test\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear DataFrames para los conjuntos de entrenamiento y prueba
    # ojo!!! cambio Type por Label
    train_df = pd.DataFrame({'text': X_train, 'label': y_train}) #'Type': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test}) #X_test, 'Type': y_test})

    # retocamos train_df, agrupandolo por tipo y tomamos muestra aleatoria de filas
    train_df = train_df.groupby("label").sample(n=max_instances_per_class, random_state=random_seed)

    # definimos el vectorizador
    # TFIDF
    #vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))
    # CountVectorizer
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))
    # HashingVectorizer
    # vectorizer = HashingVectorizer(n_features=max_features, stop_words="english", ngram_range=(1, 1))


    # vectorizamos textos de train y test
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    #X_test_f01 = vectorizer.transform(test_df_f01["text"])

    '''ruta_carpeta_xtrain= batch.functions.obtener_ruta_guardado('SaveDF', 'X_train.csv')
    ruta_carpeta_xtest = batch.functions.obtener_ruta_guardado('SaveDF', 'X_test.csv')
    ruta_carpeta_xtest_f01 = batch.functions.obtener_ruta_guardado('SaveDF', 'X_test_f01.csv')
    X_train.to_csv(ruta_carpeta_xtrain, index=False)
    X_test.to_csv(ruta_carpeta_xtest, index=False)
    X_test_f01.to_csv(ruta_carpeta_xtest_f01, index=False)'''


    # pasamos a numérico la columna label
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test = le.transform(test_df["label"])

    print("\nEligiendo calsificador")
    # Obtener una lista de todos los clasificadores disponibles
    classifiers = all_estimators(type_filter='classifier')


    # Calcular mejor algoritmo
    best_model = None
    best_score = -np.inf
    best_report = None

    # Ignorar FutureWarning y ConvergenceWarning
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # recorremos los algoritmos de clasificacion, entrenamos, medimos y actualizamos best_model con el mejor
    for name, ClassifierClass in classifiers:
        if issubclass(ClassifierClass, ClassifierMixin) and hasattr(ClassifierClass, 'fit'):
            try:

                regressor = ClassifierClass()
                print(regressor)
                regressor.fit(X_train, y_train)
                # regressor.fit(X_train.toarray(), y_train)
                y_pred = regressor.predict(X_test)
                # Calcular precision, recall, f1-score y soporte
                report = classification_report(y_test, y_pred)
                # print(y_pred)
                score = f1_score(y_test, y_pred, average="macro")
                if score > best_score:
                    best_score = score
                    best_model = regressor
                    best_report = report
                    print(f"\nModel: {name} Macro F1: {score:.3f}")
            except Exception as e:
                print(f"Error en el modelo {name}: {e}")
                continue

    # imprimimos los resultados
    print(f"\nBest Model: {best_model.__class__.__name__}")
    print(f"Macro F1 on Test Data: {best_score:.3f}")
    print(f"Best Report: {best_report}")

    # entrenamos clasificador con mejor modelo
    try:
        print(best_model)
        best_model.fit(X_train, y_train)
        print(f"Modelo: {best_model.__class__.__name__} entrenado.\nGuardando clasificador...")
    except Exception as e:
        print(f"Error al entrenar el mejor modelo: {e}")

    # guardamos clasificador y vectorizador
    print("Clasificador y vectorizador guardado en fichero")
    batch.functions.guardar_clf_vct('clf', best_model)
    batch.functions.guardar_clf_vct('vct', vectorizer)

    exit()
    # batch.module4.batchFour()

    '''
        best_model = None
        best_score = -np.inf
        best_report = None

        # Definir tu conjunto de datos
        iris = load_iris()
        X_train, X_test, y_train, y_test = iris.data, iris.data, iris.target, iris.target

        # Iterar sobre cada clasificador
        for name, ClassifierClass in classifiers:
            if issubclass(ClassifierClass, ClassifierMixin) and hasattr(ClassifierClass, 'fit'):
                try:
                    # Definir el espacio de búsqueda de hiperparámetros
                    param_grid = {
                        'n_estimators': [50, 100],
                        'max_depth': [None, 10],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }

                    # Inicializar el GridSearchCV
                    grid_search = GridSearchCV(ClassifierClass(), param_grid, cv=5, scoring='f1_macro')

                    # Entrenar el modelo con búsqueda de hiperparámetros
                    grid_search.fit(X_train, y_train)

                    # Obtener el mejor modelo y predecir en el conjunto de prueba
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test)

                    # Calcular la puntuación F1 y el informe de clasificación
                    score = f1_score(y_test, y_pred, average="macro")
                    report = classification_report(y_test, y_pred)

                    # Almacenar el mejor modelo y su rendimiento si supera el anterior
                    if score > best_score:
                        best_score = score
                        best_model = best_model
                        best_report = report

                    print(f"Model: {name} Macro F1: {score:.3f}")
                except Exception as e:
                    continue

        # Mostrar el mejor modelo y su rendimiento
        print(f"Best Model: {best_model}")
        print(f"Best Score: {best_score}")
        print(f"Best Report: {best_report}")
        '''
