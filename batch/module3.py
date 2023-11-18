# import required libraries
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from skopt import BayesSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def batchThree(dfHuman, dfIA):
    print("\n############ Ejecutando Batch 3: Clasificador #############")
    max_instances_per_class = 500
    max_features = 2000  # maximum number of features extracted for our instances
    random_seed = 777  # set random seed for reproducibility
    id2label = {0: "h", 1: "g"}

    df_union = pd.concat([dfHuman, dfIA], axis=0)

    X = df_union['Text']
    y = df_union['Type']

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear DataFrames para los conjuntos de entrenamiento y prueba
    train_df = pd.DataFrame({'Text': X_train, 'Type': y_train})
    test_df = pd.DataFrame({'Text': X_test, 'Type': y_test})

    # vectorize data: extract features from our data (from text to numeric vectors)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))
    X_train = vectorizer.fit_transform(train_df["Text"])
    X_test = vectorizer.transform(test_df["Text"])

    # vectorize labels : from text to numeric vectors
    le = LabelEncoder()
    Y_train = le.fit_transform(train_df["Type"])
    Y_test = le.transform(test_df["Type"])

    # create model
    # model = LogisticRegression()
    # model = SVC(kernel="poly")
    model = GradientBoostingClassifier()

    # train model
    model.fit(X_train, Y_train)

    # get test predictions
    predictions = model.predict(X_test)

    # evaluate predictions
    target_names = [label for idx, label in id2label.items()]
    print ("\n", model)
    print(classification_report(Y_test, predictions, target_names=target_names))

    # Calcular mejor algoritmo
    best_score = float('-inf')
    best_model = None

    for name, ClassifierClass in all_estimators(type_filter='classifier'):
        if issubclass(ClassifierClass, ClassifierMixin) and hasattr(ClassifierClass, 'fit'):
            try:
                regressor = ClassifierClass()
                regressor.fit(X_train, Y_train)
                y_pred = regressor.predict(X_test)
                score = f1_score(Y_test, y_pred, average="macro")
                if score > best_score:
                    best_score = score
                    best_model = regressor
                print(f"Model: {name} Macro F1: {score:.3f}")
            except Exception as e:
                print(f"Error en el modelo {name}: {e}")
                continue

    print(f"\nBest Model: {best_model.__class__.__name__}")
    print(f"Macro F1 on Test Data: {best_score}")

    # Seleccionamos el modelo a entrenar
    model= best_model.__class__.__name__+"()"
    model= BernoulliNB()
    print (model)
    # Entrenamos el modelo
    model.fit(X_train,y_train)

    sys.exit()
