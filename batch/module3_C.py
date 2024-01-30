# import required libraries
import pandas as pd
import batch.module4
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import batch.functions
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

# Definimos variables globales
best_score_a = -np.inf
best_score_b = -np.inf


# batch 3 - Módulo que usamos para crear los test, elegir modelo, entrenar y guardar clasificador y vectorizador
def batchThree():
    print("\n############ Ejecutando Batch 3: Clasificador - Mejores de ambas subtareas #############")
    # creamos y asignamos valor a las variables
    max_instances_per_class = 4000  # nº de instancias (ejemplos de datos) que se utilizarán por cada clase
    max_features = 20000  # nº máximo de características (atributos o variables) que se extraerán o utilizarán
    random_seed = 777  # set random seed for reproducibility

    # obtener ficheros a cargar
    # obtener ficheros a cargar
    print("\nCargando ficheros...")
    fileATrain = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTrain_A.tsv')
    fileATest = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTest_A.tsv')
    fileBTrain = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTrain_B.tsv')
    fileBTest = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTest_B.tsv')
    fileFase1 = batch.functions.obtener_ruta_guardado('SaveDF', 'DSTest_fase01.tsv')

    # creamos dataframe con datos de los ficheros
    print("\nCreando DataFrames...")
    df_train_A = pd.read_csv(fileATrain, delimiter='\t')
    df_test_A = pd.read_csv(fileATest, delimiter='\t')
    df_train_B = pd.read_csv(fileBTrain, delimiter='\t')
    df_test_B = pd.read_csv(fileBTest, delimiter='\t')
    df_fase_1 = pd.read_csv(fileFase1, delimiter='\t')

    # Balanceando fichero de train
    print("Balanceamos los ficheros de la tarea A")
    df_train_A = batch.functions.balacearDF(df_train_A)
    # df_test_A = batch.functions.balacearDF(df_test_A)
    # df_fase_1 = batch.functions.balacearDF(df_fase_1)

    print("\nPreparando datos para hacer entrenamiento y test")

    # retocamos train_df, agrupandolo por tipo y tomamos muestra aleatoria de filas
    df_train_A = df_train_A.groupby("label").sample(n=max_instances_per_class, random_state=random_seed)
    df_train_B = df_train_B.groupby("label").sample(n=max_instances_per_class, random_state=random_seed)

    # Configuraciones para TfidfVectorizer
    # analyzer:
    #   - word: cada caracteristica una palabra individual
    #   - char: Cada característica representa un n-grama de caracteres, donde n es determinado por ngram_range
    #   - char_wb: Similar a 'char', pero solo incluye n-gramas que están dentro de los límites de las palabras.
    # ngram_range: determina el rango de n-gramas que se utilizarán. Por ejemplo, (1, 1) significa solo unigramas
    #   (palabras individuales), (1, 2) significa unigramas y bigramas, y (2, 2) significa solo bigramas.
    # tfidf_options:
    #   - binary: Las frecuencias de términos son binarias (0 o 1) indicando la presencia o ausencia de un término.
    #   - use_idf: Usa la Frecuencia de Documento Inversa (IDF) para ponderar las frecuencias de términos.
    #       Las palabras raras tendrán un peso más alto.
    #   - smooth_idf: Similar a 'use_idf', pero con un término de suavizado en el denominador para evitar divisiones
    #       por cero.
    #   - sublinear_tf: Aplica una escala logarítmica a las frecuencias de términos antes de aplicar TF-IDF.
    #       Esto puede ayudar a manejar mejor la varianza en las frecuencias de términos.

    # SUBTAREA A
    # Completo -BernoulliNB 0.675 - Vectorizador _char_two-two_binary
    # 441 - BernoulliNB 0.653 - Vectorizador char_wb_two-two_binary
    # 150 - SVC 0.575 - Vectorizador _word_two-two_binary
    # 50 - RandomForestClassifier 0.583 - Vectorizador char_one-one_binary
    # SUBTAREA B
    # Completo -
    # 441 - GradientBoostingClassifier 0.502 - Vectorizador char_two-two_use_idf
    # 150 - GradientBoostingClassifier 0.414 - Vectorizador char_wb_two-two_binary
    # 50 - AdaBoostClassifier 0.299 - Vectorizador char_wb_two-two_use_idf

    # Creamos Vectorizadores
    v_A_completo = TfidfVectorizer(max_features=max_features, analyzer='char', ngram_range=(2, 2), use_idf=True)
    v_A_441 = TfidfVectorizer(max_features=max_features, analyzer='char_wb', ngram_range=(2, 2))
    v_A_150 = TfidfVectorizer(max_features=max_features, analyzer='word', ngram_range=(2, 2), stop_words="english")
    v_A_50 = TfidfVectorizer(max_features=max_features, analyzer='char', ngram_range=(1, 1))
    v_B_completo = TfidfVectorizer(max_features=max_features, analyzer='char', ngram_range=(2, 2), use_idf=True)
    v_B_441 = TfidfVectorizer(max_features=max_features, analyzer='char', ngram_range=(2, 2), use_idf=True)
    v_B_150 = TfidfVectorizer(max_features=max_features, analyzer='char_wb', ngram_range=(2, 2))
    v_B_50 = TfidfVectorizer(max_features=max_features, analyzer='char_wb', ngram_range=(2, 2), use_idf=True)

    # Creamos Clasificadores
    c_A_completo = BernoulliNB()
    c_A_441 = BernoulliNB()
    c_A_150 = svm.SVC(kernel='linear')
    c_A_50 = RandomForestClassifier(n_estimators=150, random_state=42)
    c_B_completo = GradientBoostingClassifier(n_estimators=100, random_state=42)
    c_B_441 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    c_B_150 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    c_B_50 = AdaBoostClassifier(n_estimators=50, random_state=42)

    # Definimos tipos de columna a usar
    txt_completo = "text"
    txt_441 = "tokenized_text"
    txt_150 = "tokenized_text_150"
    txt_50 = "tokenized_text_50"

    # Creamos dataframe para completar los datos de todos los modelos
    column_names = ['subtarea', 'columna', 'modelo', 'model', 'score', 'report', 'score_f01', 'report_f01']
    df_total = pd.DataFrame(columns=column_names)

    def comprobar_sistema(vectorizer, clasificador, df_train, df_test, columna, subtarea, nombre_mostrar):
        global best_score_a, best_score_b

        print("\nCargando vectorizador para Subtarea ", subtarea, " usando la columna ", columna)
        X_train = vectorizer.fit_transform(df_train[columna])
        X_test = vectorizer.transform(df_test[columna])
        X_test_f01 = vectorizer.transform(df_fase_1[columna])

        le = LabelEncoder()
        y_train = le.fit_transform(df_train["label"])
        y_test = le.transform(df_test["label"])
        y_test_f01 = le.transform(df_fase_1["label"])

        try:
            clasificador.fit(X_train, y_train)
            y_pred = clasificador.predict(X_test)
            y_pred_f01 = clasificador.predict(X_test_f01)
            # Calcular precision, recall, f1-score y soporte
            report = classification_report(y_test, y_pred)
            report_f01 = classification_report(y_test_f01, y_pred_f01)
            # Calculamos el score
            score = f1_score(y_test, y_pred, average="macro", zero_division=1)
            score_f01 = f1_score(y_test_f01, y_pred_f01, average="macro", zero_division=1)
            new_row = [subtarea, columna, clasificador.__class__.__name__, clasificador, score, report, report_f01,
                       score_f01]
            df_total.loc[len(df_total)] = new_row
            # Guardamos tabla con valores de todos los entrenamientos hasta ahora.  Pisamos la anterior
            df_total.to_csv(batch.functions.obtener_ruta_guardado('Estadisticas', 'TODOS_tabla_mejoresModulo3C.tsv'))
        except Exception as e:
            print(f"Error : {e}")
            # guardamos clasificador y vectorizador
        if subtarea == 'A':
            if score > best_score_a:
                best_score_a = score
                print("\nGuardando clasificador, mejor de subtarea A de momento, con f1_score = ", score)
                batch.functions.guardar_clf_vct('clf', clasificador, 'A')
                batch.functions.guardar_clf_vct('vct', vectorizer, 'A')
        else:
            if score > best_score_b:
                best_score_b = score
                print("\nGuardando clasificador, mejor de subtarea B de momento, con f1_score = ", score)
                batch.functions.guardar_clf_vct('clf', clasificador, 'B')
                batch.functions.guardar_clf_vct('vct', vectorizer, 'B')

        print("\nGuardando clasificador...")
        batch.functions.guardar_clf_vct_nombre('clf', clasificador, subtarea, nombre_mostrar)
        batch.functions.guardar_clf_vct_nombre('vct', vectorizer, subtarea, nombre_mostrar)
        print("\nClasificador y vectorizador guardado en fichero")

    # Datos subtarea A
    comprobar_sistema(v_A_completo, c_A_completo, df_train_A, df_test_A, txt_completo, 'A', 'S_A_completo')
    comprobar_sistema(v_A_441, c_A_441, df_train_A, df_test_A, txt_441, 'A', 'S_A_441')
    comprobar_sistema(v_A_150, c_A_150, df_train_A, df_test_A, txt_150, 'A', 'S_A_150')
    comprobar_sistema(v_A_50, c_A_50, df_train_A, df_test_A, txt_50, 'A', 'S_A_50')

    # Datos subtarea B
    comprobar_sistema(v_B_completo, c_B_completo, df_train_B, df_test_B, txt_completo, 'B', 'S_B_completo')
    comprobar_sistema(v_B_441, c_B_441, df_train_B, df_test_B, txt_441, 'B', 'S_B_441')
    comprobar_sistema(v_B_150, c_B_150, df_train_B, df_test_B, txt_150, 'B', 'S_B_150')
    comprobar_sistema(v_B_50, c_B_50, df_train_B, df_test_B, txt_50, 'B', 'S_B_50')

    # Guardamos tabla con valores de todos los entrenamientos
    df_total.to_csv(batch.functions.obtener_ruta_guardado('Estadisticas', 'TODOS_tabla_mejoresModulo3C.tsv'))

    batch.module4.batchFour(" ", " ")
