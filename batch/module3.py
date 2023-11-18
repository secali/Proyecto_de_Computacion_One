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

    x = df_union['Text']
    y = df_union['Type']


    # Dividir los datos en conjuntos de entrenamiento y prueba
    train, test= train_test_split(x, y, test_size=0.2, random_state=42)


    # vectorize data: extract features from our data (from text to numeric vectors)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))
    X_train = vectorizer.fit_transform(train["Text"])
    X_test = vectorizer.transform(test["Text"])
