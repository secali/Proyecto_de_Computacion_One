# import required libraries
import batch.functions
import pandas as pd
import batch.module3_B
import batch.module3_A


# batch 2 - Módulo que usamos para cargar ficheros, preparar los datos e imprimir estadísticas
def batchTwo():
    print("\n############ Ejecutando Batch 2: carga ficheros, estadística y manejo de datos #############")

    # Obtenemos ruta de ficheros a cargar
    print("\nCargando ficheros...")
    fileATrain = batch.functions.obtener_ruta_guardado('Descargas', 'subtaskA_train_monolingual.jsonl')
    fileATest = batch.functions.obtener_ruta_guardado('Descargas', 'subtaskA_dev_monolingual.jsonl')
    fileBTrain = batch.functions.obtener_ruta_guardado('Descargas', 'subtaskB_train.jsonl')
    fileBTest = batch.functions.obtener_ruta_guardado('Descargas', 'subtaskB_dev.jsonl')
    fileFase1 = batch.functions.obtener_ruta_guardado('Descargas', 'DataSetFinal.tsv')

    # Creamos dataframe de los ficheros que obtuvimos ruta, cada uno con su tipo.
    print("\nCreando DataFrames...")
    df_train_A = pd.read_json(fileATrain, lines=True)
    df_test_A = pd.read_json(fileATest, lines=True)
    df_train_B = pd.read_json(fileBTrain, lines=True)
    df_test_B = pd.read_json(fileBTest, lines=True)
    df_fase_1 = pd.read_csv(fileFase1, delimiter='\t')

    # Actualizamos nombre columnas df_fase_1 para estandarizarlas con el resto de dataframes
    print("\nActualizando datos DataFrame fase_01...")
    df_fase_1.columns = ['text', 'label']
    df_fase_1['label'].replace({'h': 0, 'g': 1}, inplace=True)

    print("\nPreparando los datos...")
    #   Preparamos datosVamos a tratar en todos los ficheros:
    #   - Limpiar texto < de 20 caracteres
    #   - Limpiar tabulaciones, saltos de línea y dobles espacios
    #   - Limpiar texto que no esté en inglés.
    #   - Eliminar filas duplicadas.
    #   - Eliminar filas con alguna columna vacía.
    #   - Tokenizer a 441 tokens, 50 tokens y 150 tokens, en columnas diferentes

    # Tratamos datos de Train A
    batch.functions.guardar_dataset(df_train_A, 'DSTrain_A_sucio.tsv')
    df_train_A = batch.functions.limpia_texto_df(df_train_A)
    batch.functions.guardar_dataset(df_train_A, 'DSTrain_A.tsv')

    # Tratamos datos de Test A
    batch.functions.guardar_dataset(df_test_A, 'DSTest_A_sucio.tsv')
    df_test_A = batch.functions.limpia_texto_df(df_test_A)
    batch.functions.guardar_dataset(df_test_A, 'DSTest_A.tsv')

    # Tratamos datos de Train B
    batch.functions.guardar_dataset(df_train_B, 'DSTrain_B_sucio.tsv')
    df_train_B = batch.functions.limpia_texto_df(df_train_B)
    batch.functions.guardar_dataset(df_train_B, 'DSTrain_B.tsv')

    # Tratamos datos de Test B
    batch.functions.guardar_dataset(df_test_B, 'DSTest_B_sucio.tsv')
    df_test_B = batch.functions.limpia_texto_df(df_test_B)
    batch.functions.guardar_dataset(df_test_B, 'DSTest_B.tsv')

    # Tratamos datos fichero fase 01
    batch.functions.guardar_dataset(df_fase_1, 'DSTest_fase01_sucio.tsv')
    df_fase_1 = batch.functions.limpia_texto_df(df_fase_1)
    batch.functions.guardar_dataset(df_fase_1, 'DSTest_fase01.tsv')

    # Imprimimos estadísticas y las guardamos en /Estadísticas
    print("\nImprimiendo y guardando estadísticas...")
    batch.functions.imprime_estadistica(df_train_A, 'Estadísticas Subtask A Train', 'E_SubtaskA_Modulo02_Train.tsv')
    batch.functions.imprime_estadistica(df_test_A, 'Estadísticas Subtask A Test', 'E_SubtaskA_Modulo02_Test.tsv')
    batch.functions.imprime_estadistica(df_train_B, 'Estadísticas Subtask B Train', 'E_SubtaskB_Modulo02_Train.tsv')
    batch.functions.imprime_estadistica(df_test_B, 'Estadísticas Subtask B Test', 'E_SubtaskB_Modulo02_Test.tsv')
    batch.functions.imprime_estadistica(df_fase_1, 'Estadísticas Data Frame Fase 01', 'E_Fase01_Modulo02_Test.tsv')

    # Pasamos al módulo 3, para hacer el tratamiento de los datos
    batch.module3_A.batchThree()
