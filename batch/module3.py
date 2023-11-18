# import required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def batchThree(dfHuman, dfIA):
    print("\n############ Ejecutando Batch 3: Clasificador #############")
    max_instances_per_class = 500
    max_features = 2000  # maximum number of features extracted for our instances
    random_seed = 777  # set random seed for reproducibility
    id2label = {0: "h", 1: "g"}

    df_union = pd.concat([dfHuman, dfIA], axis=0)

    X = df_union.drop('Type', axis=1)
    y = df_union['Type']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


