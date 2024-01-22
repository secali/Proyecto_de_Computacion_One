# import required libraries
import sys

import pandas as pd
import warnings
import batch.module4
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
import nltk
from nltk.tokenize import word_tokenize

# batch 3 - modulo que usamos para crear los test, elegir modelo, entrenar y guardar clasificador y vectorizador
def batchThree():
    print("\n############ Ejecutando Batch 3: Clasificador - Subtarea B #############")
    # creamos y asignamos valor a las variables
    max_instances_per_class = 1000 #100  # numero de instancias por clase
    max_features = 500  # maximum number of features extracted for our instances
    random_seed = 777  # set random seed for reproducibility

    # obtener ficheros a cargar
    print("\nCargando ficheros...")
    fileATrain = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTrain_B.tsv')
    fileATest = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTest_B.tsv')
    # fileFase1 = batch.functions.obtener_ruta_guardado('SaveDF', 'DataSetFinal.tsv')
    fileFase1 = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTest_fase01.tsv')

    # creamos dataframe con datos de los ficheros
    print("\nCreando DataFrames...")
    df_train_B = pd.read_csv(fileATrain, delimiter='\t')
    df_test_B = pd.read_csv(fileATest, delimiter='\t')
    df_fase_1 = pd.read_csv(fileFase1, delimiter='\t')

    # Imrpimimos estadistica
    batch.functions.imprime_estadistica_subtarea_A(df_train_B, df_test_B, df_fase_1)



    # determine avg text length in tokens
    num=int(df_train_B["text"].map(lambda x: len(x.split(" "))).mean())
    print("Numero de caracteres al que reducimos el texto",num)


    # Imrpimimos estadistica
    print("\nEstadistica con fichero balanceado")
    batch.functions.imprime_estadistica_subtarea_B(df_train_B, df_test_B, df_fase_1)

    print("\nPreparando datos para hacer entrenamiento y test")

    # Crear DataFrames para los conjuntos de entrenamiento y prueba
    train_df = df_train_B
    test_df = df_test_B
    test_df_f01 = df_fase_1


    # retocamos train_df, agrupandolo por tipo y tomamos muestra aleatoria de filas
    train_df = train_df.groupby("label").sample(n=max_instances_per_class, random_state=random_seed)

    # definimos el vectorizador
    # TFIDF
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))
    # CountVectorizer
    #vectorizer = CountVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))
    # HashingVectorizer
    # vectorizer = HashingVectorizer(n_features=max_features, stop_words="english", ngram_range=(1, 1))


    # vectorizamos textos de train y test
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    #X_train = vectorizer.fit_transform(train_df["tokenized_text"])
    #X_test = vectorizer.transform(test_df["tokenized_text"])
    #X_test_f01 = vectorizer.transform(test_df_f01["text"])



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
    batch.functions.guardar_clf_vct('clf_B', best_model)
    batch.functions.guardar_clf_vct('vct_B', vectorizer)

    batch.module4.batchFour("", "")
