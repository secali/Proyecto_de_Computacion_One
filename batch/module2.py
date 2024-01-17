# import required libraries
import batch.functions
import pandas as pd
import batch.module3
import nltk


# batch 2 - modulo que usamos para cargar fichero, hacer estadísticas y balancear los datos
def batchTwo(modelo, texto):
    print("\n############ Ejecutando Batch 2: carga ficheros, estadística y manejo de datos #############")
    # obtener ficheros a cargar
    print("\nCargando ficheros...")
    fileATrain = batch.functions.obtener_ruta_guardado('Descargas', 'subtaskA_train_monolingual.jsonl')
    fileATest = batch.functions.obtener_ruta_guardado('Descargas', 'subtaskA_dev_monolingual.jsonl')
    fileBTrain = batch.functions.obtener_ruta_guardado('Descargas', 'subtaskB_train.jsonl')
    fileBTest = batch.functions.obtener_ruta_guardado('Descargas', 'subtaskB_dev.jsonl')
    fileFase1 = batch.functions.obtener_ruta_guardado('SaveDF', 'DataSetFinal.tsv')

    # creamos dataframe con datos fichero
    print("\nCreando DataFrames...")
    df_train_A = pd.read_json(fileATrain, lines=True)
    df_test_A = pd.read_json(fileATest, lines=True)
    df_train_B = pd.read_json(fileBTrain, lines=True)
    df_test_B = pd.read_json(fileBTest, lines=True)
    df_fase_1 = pd.read_csv(fileFase1, delimiter='\t')

    # Actualizamos nombre columnas df_fase_1 para estandarizarlas con el resto
    print("\nActualizando datos DataFrame fase_01...")
    df_fase_1.columns = ['text', 'label']
    df_fase_1['label'].replace({'h': 0, 'g': 1}, inplace=True)

    # Preparamos datos
    nltk.download('punkt')
    print("\nPreparando los datos...")
    batch.functions.guardar_dataset(df_train_A, 'DSTrain_A_sucio.tsv')
    df_train_A = batch.functions.limpia_texto_df(df_train_A)
    df_train_A['tokenized_text'] = df_train_A['text'].apply(batch.functions.tokenize_and_reduce)
    batch.functions.guardar_dataset(df_train_A, 'DSTrain_A.tsv')

    batch.functions.guardar_dataset(df_test_A, 'DSTest_A_sucio.tsv')
    df_test_A = batch.functions.limpia_texto_df(df_test_A)
    df_test_A['tokenized_text'] = df_test_A['text'].apply(batch.functions.tokenize_and_reduce)
    batch.functions.guardar_dataset(df_test_A, 'DSTest_A.tsv')

    batch.functions.guardar_dataset(df_train_B, 'DSTrain_B_sucio.tsv')
    df_train_B = batch.functions.limpia_texto_df(df_train_B)
    df_train_B['tokenized_text'] = df_train_B['text'].apply(batch.functions.tokenize_and_reduce)
    batch.functions.guardar_dataset(df_train_B, 'DSTrain_B.tsv')

    batch.functions.guardar_dataset(df_test_B, 'DSTest_B_sucio.tsv')
    df_test_B = batch.functions.limpia_texto_df(df_test_B)
    df_test_B['tokenized_text'] = df_test_B['text'].apply(batch.functions.tokenize_and_reduce)
    batch.functions.guardar_dataset(df_test_B, 'DSTest_B.tsv')

    batch.functions.guardar_dataset(df_fase_1, 'DSTest_fase01_largo.tsv')
    df_fase_1['tokenized_text'] = df_fase_1['text'].apply(batch.functions.tokenize_and_reduce)
    batch.functions.guardar_dataset(df_fase_1, 'DSTest_fase01.tsv')


    # Imprimimos estadisticas
    print("\nImprimiendo estadísticas...")
    batch.functions.imprime_estadistica(df_train_A, 'Estadísticas Subtask A Train')
    batch.functions.imprime_estadistica(df_test_A, 'Estadísticas Subtask A Test')
    batch.functions.imprime_estadistica(df_train_B, 'Estadísticas Subtask B Train')
    batch.functions.imprime_estadistica(df_test_B, 'Estadísticas Subtask B Test')
    batch.functions.imprime_estadistica(df_fase_1, 'Estadísticas Data Frame Fase 01')


    #print(df_train_A)
    #print(df_test_A)
    #print(df_train_B)
    #print(df_test_B)
    #print(df_fase_1)
    '''
    # Limpiar Texto, eliminar cadenas vacias, eliminar duplicados
    print("\nLimpiando texto...")
    print("\nTrain A...")
    df_train_A=batch.functions.limpia_texto_df(df_train_A)
    #df_train_A['text'] = df_train_A['text'].apply(batch.functions.limpiar_texto)
    df_train_A = df_train_A[df_train_A['text'] != '']
    df_train_A = df_train_A.drop_duplicates()
    df_train_A = df_train_A.dropna(subset=['text'])
    df_train_A = df_train_A.dropna(subset=['label'])
    batch.functions.guardar_dataset(df_train_A,'DSTrain_A.tsv')
    print("\nTest A...")
    df_test_A = batch.functions.limpia_texto_df(df_test_A)
    # df_test_A['text'] = df_test_A['text'].apply(batch.functions.limpiar_texto)
    df_test_A = df_test_A[df_test_A['text'] != '']
    df_test_A = df_test_A.drop_duplicates()
    df_test_A = df_test_A.dropna(subset=['text'])
    df_test_A = df_test_A.dropna(subset=['label'])
    batch.functions.guardar_dataset(df_test_A, 'DSTest_A.tsv')
    print("\nTrain B...")
    df_train_B = batch.functions.limpia_texto_df(df_train_B)
    # df_train_B['text'] = df_train_B['text'].apply(batch.functions.limpiar_texto)
    df_train_B = df_train_B[df_train_B['text'] != '']
    df_train_B = df_train_B.drop_duplicates()
    df_train_B = df_train_B.dropna(subset=['text'])
    df_train_B = df_train_B.dropna(subset=['label'])
    batch.functions.guardar_dataset(df_train_B, 'DSTrain_B.tsv')
    print("\nTest B...")
    df_test_B = batch.functions.limpia_texto_df(df_test_B)
    # df_test_B['text'] = df_test_B['text'].apply(batch.functions.limpiar_texto)
    df_test_B = df_test_B[df_test_B['text'] != '']
    df_test_B = df_test_B.drop_duplicates()
    df_test_B = df_test_B.dropna(subset=['text'])
    df_test_B = df_test_B.dropna(subset=['label'])
    batch.functions.guardar_dataset(df_test_B, 'DSTest_B.tsv')
    print("\nTest fase 01...")
    batch.functions.guardar_dataset(df_fase_1, 'DSTest_fase01.tsv')


    # Imprimimos estadisticas
    print("\nImprimiendo estadísticas...")
    batch.functions.imprime_estadistica(df_train_A, 'Estadísticas Subtask A Train')
    batch.functions.imprime_estadistica(df_test_A, 'Estadísticas Subtask A Test')
    batch.functions.imprime_estadistica(df_train_B, 'Estadísticas Subtask B Train')
    batch.functions.imprime_estadistica(df_test_B, 'Estadísticas Subtask B Test')
    batch.functions.imprime_estadistica(df_fase_1, 'Estadísticas Data Frame Fase 01')

    # Balancear resultados
    # df_train_A = batch.functions.balacearDF(df_train_A)
    # df_train_B = batch.functions.balacearDF(df_train_B)

    # guardamos los dataset finales balanceados, por si tuvieramos que recuperarlos
    # batch.functions.guardar_dataset(df_train_A, 'df_train_A.tsv')
    # batch.functions.guardar_dataset(df_test_A, 'df_test_A.tsv')
    # batch.functions.guardar_dataset(df_train_B, 'df_train_B.tsv')
    # batch.functions.guardar_dataset(df_test_B, 'df_test_B.tsv')
    # batch.functions.guardar_dataset(df_fase_1, 'df_fase_1.tsv')
'''
    batch.module3.batchThree(modelo, texto)
    #batch.module4.batchFour(df_train_B, df_test_B, df_fase_1)


