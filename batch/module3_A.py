# import required libraries
import sys

import inflect
import pandas as pd
import warnings
import batch.module4
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
import batch.functions
import batch.module3_B
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm


# batch 3 - Módulo que usamos para crear los test, elegir modelo, entrenar y guardar clasificador y vectorizador
def batchThree():
    print("\n############ Ejecutando Batch 3: Clasificador - Subtarea A #############")
    # creamos y asignamos valor a las variables
    max_instances_per_class = 1000  # 1000  # 100  # numero de instancias por clase
    max_features = 5000  # 5000  # maximum number of features extracted for our instances
    random_seed = 777  # set random seed for reproducibility

    # obtener ficheros a cargar
    print("\nCargando ficheros...")
    fileATrain = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTrain_A.tsv')
    fileATest = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTest_A.tsv')
    fileFase1 = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTest_fase01.tsv')

    # creamos dataframe con datos de los ficheros
    print("\nCreando DataFrames...")
    df_train_A = pd.read_csv(fileATrain, delimiter='\t')
    df_test_A = pd.read_csv(fileATest, delimiter='\t')
    df_fase_1 = pd.read_csv(fileFase1, delimiter='\t')

    # Imrpimimos estadistica
    batch.functions.imprime_estadistica_subtarea_A(df_train_A, df_test_A, df_fase_1,
                                                   'E_SubtaskA_FicheroNoBalanceado.tsv')

    # Balanceando fichero de train
    print("Balanceamos los ficheros")
    df_train_A = batch.functions.balacearDF(df_train_A)
    df_test_A = batch.functions.balacearDF(df_test_A)
    df_fase_1 = batch.functions.balacearDF(df_fase_1)

    # determine avg text length in tokens
    num = int(df_train_A["text"].map(lambda x: len(x.split(" "))).mean())
    print("Numero de caracteres al que reducimos el texto", num)

    # Imrpimimos estadistica
    print("\nEstadistica con fichero balanceado")
    batch.functions.imprime_estadistica_subtarea_A(df_train_A, df_test_A, df_fase_1, 'E_SubtaskA_FicheroBalanceado.tsv')

    print("\nPreparando datos para hacer entrenamiento y test")

    # Crear DataFrames para los conjuntos de entrenamiento y prueba
    train_df = df_train_A  # pd.DataFrame({'text': X_train, 'label': y_train})  # 'Type': y_train})
    test_df = df_test_A  # pd.DataFrame({'text': X_test, 'label': y_test})  # X_test, 'Type': y_test})
    test_df_f01 = df_fase_1  # pd.DataFrame({'text': X_test_f01, 'label': y_test_f01})  # X_test, 'Type': y_test})'''

    # retocamos train_df, agrupandolo por tipo y tomamos muestra aleatoria de filas
    train_df = train_df.groupby("label").sample(n=max_instances_per_class, random_state=random_seed)

    # definimos el vectorizador
    # TFIDF
    # vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))

    # CountVectorizer
    # vectorizer = CountVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 1))
    # HashingVectorizer
    # vectorizer = HashingVectorizer(n_features=max_features, stop_words="english", ngram_range=(1, 1))
    # Pasamos a numérico la columna label
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test = le.transform(test_df["label"])
    y_test_f01 = le.transform(test_df_f01["label"])

    def ngram_range_to_text(ngram_range):
        p = inflect.engine()
        start, end = ngram_range
        return f"{p.number_to_words(start)}-gram to {p.number_to_words(end)}-gram"

    # Configuraciones para TfidfVectorizer
    analyzer_options = ['word', 'char', 'char_wb']
    ngram_ranges = [(1, 1), (1, 2), (2, 2)]  # Experimenta con diferentes valores
    tfidf_options = ['binary', 'use_idf', 'smooth_idf', 'sublinear_tf']

    best_score_group_t = -np.inf
    best_model_group_t = None
    best_report_group_t = None
    best_analyzer_t = None
    best_ngram_ranges_t = [(1, 1)]
    best_tfidf_options_t = None
    best_score_group_n = -np.inf
    best_model_group_n = None
    best_report_group_n = None
    best_analyzer_n = None
    best_ngram_ranges_n = [(1, 1)]
    best_tfidf_options_n = None
    best_score_group_a = -np.inf
    best_model_group_a = None
    best_report_group_a = None
    best_analyzer_a = None
    best_ngram_ranges_a = [(1, 1)]
    best_tfidf_options_a = None

    # Iterar sobre las configuraciones
    for analyzer in analyzer_options:
        for ngram_range in ngram_ranges:
            for tfidf_option in tfidf_options:
                # Configurar el TfidfVectorizer
                vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                             use_idf=(tfidf_option == 'use_idf'),
                                             smooth_idf=(tfidf_option == 'smooth_idf'),
                                             sublinear_tf=(tfidf_option == 'sublinear_tf'))

                # Aplicar la transformación TF-IDF a los datos
                # vectorizamos textos de train y test
                X_train = vectorizer.fit_transform(train_df["text"])
                X_test = vectorizer.transform(test_df["text"])
                # X_train = vectorizer.fit_transform(train_df["tokenized_text_50"])
                # X_test = vectorizer.transform(test_df["tokenized_text_50"])
                # X_test_f01 = vectorizer.transform(test_df_f01["tokenized_text_50"])
                # X_train = vectorizer.fit_transform(train_df["tokenized_text_150"])
                # X_test = vectorizer.transform(test_df["tokenized_text_150"])
                # X_test_f01 = vectorizer.transform(test_df_f01["tokenized_text_150"])
                # X_train = vectorizer.fit_transform(train_df["tokenized_text"])
                # X_test = vectorizer.transform(test_df["tokenized_text"])
                # X_test_f01 = vectorizer.transform(test_df_f01["tokenized_text"])

                # Imprimir información sobre la configuración actual
                #  print(f"\nConfiguration: Analyzer={analyzer}, Ngram Range={ngram_range}, TF-IDF Option={tfidf_option}")
                # print(f"Number of features: {len(vectorizer.get_feature_names())}")
                # print(f"Example feature names: {vectorizer.get_feature_names()[:5]}")
                # print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

                print("\nEligiendo calsificador")
                # Obtener una lista de todos los clasificadores disponibles
                classifiers = all_estimators(type_filter='classifier')
                # print (classifiers)

                # Calcular mejor algoritmo
                best_model = None
                best_score = -np.inf
                best_report = None

                # Ignorar FutureWarning y ConvergenceWarning
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                # recorremos los algoritmos de clasificacion, entrenamos, medimos y actualizamos best_model con el mejor
                total_filas = len(classifiers)
                pbar = tqdm(total=total_filas)  # Inicializa la barra de progreso
                for name, ClassifierClass in classifiers:
                    if issubclass(ClassifierClass, ClassifierMixin) and hasattr(ClassifierClass, 'fit'):
                        try:

                            regressor = ClassifierClass()
                            # print(regressor)
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
                                # print(f"\nModel: {name} Macro F1: {score:.3f}")
                        except Exception as e:
                            # print(f"Error en el modelo {name}: {e}")
                            continue
                    # Actualiza la barra de progreso
                    pbar.update(1)
                # Cierra la barra de progreso al finalizar
                pbar.close()

                # Imprimimos los resultados y los guardamos en fichero
                print(f"\nBest Model: {best_model.__class__.__name__}")
                print(f"Macro F1 on Test Data: {best_score:.3f}")
                print(f"Best Report: {best_report}")
                batch.functions.guardar_report(best_model.__class__.__name__, best_score, best_report,
                                               "SubtareaA_Modulo3A_MejorModelo_" + analyzer + "_"
                                               + ngram_range_to_text(ngram_range) + "_" + tfidf_option + ".tsv")
                if best_score_group_t < best_score:
                    best_score_group_t = best_score
                    best_model_group_t = best_model
                    best_report_group_t = best_report
                    best_analyzer_t = analyzer
                    best_ngram_ranges_t = ngram_range
                    best_tfidf_options_t = tfidf_option
            if best_score_group_n < best_score_group_t:
                best_score_group_n = best_score_group_t
                best_model_group_n = best_model_group_t
                best_report_group_n = best_report_group_t
                best_analyzer_n = best_analyzer_t
                best_ngram_ranges_n = best_ngram_ranges_t
                best_tfidf_options_n = best_tfidf_options_t
        if best_score_group_a < best_score_group_n:
            best_score_group_a = best_score_group_n
            best_model_group_a = best_model_group_n
            best_report_group_a = best_report_group_n
            best_analyzer_a = best_analyzer_n
            best_ngram_ranges_a = best_ngram_ranges_n
            best_tfidf_options_a = best_tfidf_options_n
    # Imprimimos los resultados y los guardamos en fichero
    print(f"\nBest Model: {best_model_group_a.__class__.__name__}")
    print(f"Macro F1 on Test Data: {best_score_group_a:.3f}")
    print(f"Best Report: {best_report_group_a}")
    batch.functions.guardar_report(best_model_group_a.__class__.__name__, best_score_group_a, best_report_group_a,
                                   best_analyzer_a + "_"
                                   + ngram_range_to_text(best_ngram_ranges_a) + "_" + best_tfidf_options_a + "_"
                                   + "SubtareaA_Modulo3A_MejorModelo_FINAL.tsv")
    mejor_vectorizer = TfidfVectorizer(analyzer=best_analyzer_a, ngram_range=best_ngram_ranges_a,
                                       use_idf=(best_tfidf_options_a == 'use_idf'),
                                       smooth_idf=(best_tfidf_options_a == 'smooth_idf'),
                                       sublinear_tf=(best_tfidf_options_a == 'sublinear_tf'))

    X_train = mejor_vectorizer.fit_transform(train_df["text"])
    X_test_f01 = mejor_vectorizer.transform(test_df_f01["text"])
    # Testeamos modelo fase 1 e imprimimos report
    try:
        print("\nProbamos con test de fase 01 el modelo ", best_model_group_a)
        best_model = best_model_group_a.fit(X_train, y_train)
        y_pred = best_model.predict(X_test_f01)
        # Calcular precision, recall, f1-score y soporte
        report = classification_report(y_test_f01, y_pred)
        score = f1_score(y_test_f01, y_pred, average="macro")
        # Imprimimos los resultados
        print(f"\nBest Model: {best_model.__class__.__name__}")
        print(f"Macro F1 on Test Data: {score:.3f}")
        print(f"Best Report: {report}")
        batch.functions.guardar_report(best_model.__class__.__name__, report, score,
                                       'SubtareaA_Modulo3A_PruebaTestFase01.tsv')
    except Exception as e:
        print(f"Error al testear con : {e}")

    # entrenamos clasificador con mejor modelo
    try:
        print("\nEntrenamos clasificador con modelo ", best_model_group_a)
        best_model = best_model_group_a.fit(X_train, y_train)
        print(f"Modelo: {best_model.__class__.__name__} entrenado.")
    except Exception as e:
        print(f"Error al entrenar el mejor modelo: {e}")

    # guardamos clasificador y vectorizador
    print("\nGuardando clasificador...")
    batch.functions.guardar_clf_vct('clf', best_model_group_a, 'A')
    batch.functions.guardar_clf_vct('vct', mejor_vectorizer, 'A')
    print("\nClasificador y vectorizador guardado en fichero")

    # batch.module3_B.batchThree()

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
