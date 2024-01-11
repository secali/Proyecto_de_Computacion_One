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
    # Supongamos que tienes datos de entrenamiento X_train y las etiquetas correspondientes y_train
    # Además, tienes datos de prueba X_test y las etiquetas correspondientes y_test

    # Separar las características (X) del objetivo (y)
    X = df_train_A['text']
    y = df_train_A['label']

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    print("Creando ficheros de entranamiento y test\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectoriza los textos utilizando CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Inicializa y entrena el clasificador Naive Bayes
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vec, y_train)

    # Realiza predicciones en el conjunto de prueba
    y_pred = nb_classifier.predict(X_test_vec)

    # Evalúa el rendimiento del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:\n', report)