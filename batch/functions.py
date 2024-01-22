import numpy as np

import sys

import batch

sys.path.append("../batch")
from batch import module1
from batch import module2
from batch import module3_A
from batch import module3_B
from batch import module4
import os
import datetime
from joblib import dump
from bs4 import BeautifulSoup
from langdetect import detect
import gdown
from tqdm import tqdm
from tabulate import tabulate
import pandas as pd
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


# Function to tokenize and reduce to 50 words
def tokenize_and_reduce(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens[:441])


def tokenize_and_reduce_50(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens[:50])


def tokenize_and_reduce_150(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens[:150])


def limpia_texto(list):
    lista_txt_limpio = []
    for text in list:
        if len(text.get_text()) >= 20 and detect(text.get_text()) == 'en':
            lista_txt_limpio.append(text.get_text().replace('\t', '').replace('\n', ''))
    return lista_txt_limpio


def obtener_datos_2():
    # global ruta_script, ruta_carpeta, file_01, file02
    # obtener ruta si existe dataset DataSet - format TSV
    # Tomamos los ultimos archivos que se cargan y se procesan, para ver si esta realizada la descarga
    file_01 = obtener_ruta_guardado('Descargas', 'subtaskB_train.jsonl')
    file_02 = obtener_ruta_guardado('SaveDF', 'DSTest_B.tsv')
    file_03_01 = obtener_ruta_guardado('SaveCLF', 'clf_B.joblib')
    file_03_02 = obtener_ruta_guardado('SaveVCT', 'vct_B.joblib')
    flag_tratamiento = 0

    if (os.path.exists(file_03_01)) and (os.path.exists(file_03_02)):
        # Obtener la fecha de modificación del archivo
        fecha_modificacion = datetime.datetime.fromtimestamp(os.path.getmtime(file_03_01))
        # Obtener la fecha de creación del archivo (solo disponible en algunos sistemas)
        flag_tratamiento = 3
        # Mostrar las fechas
        print(f"\nYa se ha seleccionado el mejor clasificador y el vectorizador."
              f"\nFecha de modificación: {fecha_modificacion}")
        print("Si desea continuar con el proceso sin tocar los datos anteriores seleccione la opcion 4")
    elif os.path.exists(file_02):
        # Obtener la fecha de modificación del archivo
        fecha_modificacion = datetime.datetime.fromtimestamp(os.path.getmtime(file_02))
        # Obtener la fecha de creación del archivo (solo disponible en algunos sistemas)
        flag_tratamiento = 2
        # Mostrar las fechas
        print(f"\nYa se ha realizado un procesamiento de datos."
              f"\nFecha de modificación: {fecha_modificacion}")
        print("Si desea continuar con el proceso sin tocar los datos anteriores seleccione la opcion 3")
    elif os.path.exists(file_01):
        # Obtener la fecha de modificación del archivo
        fecha_modificacion = datetime.datetime.fromtimestamp(os.path.getmtime(file_01))
        # Obtener la fecha de creación del archivo (solo disponible en algunos sistemas)
        flag_tratamiento = 1
        # Mostrar las fechas
        print(f"\nYa se ha realizado una descarga anterior."
              f"\nFecha de descarga: {fecha_modificacion}")
        print("Si desea continuar con el proceso sin tocar los datos anteriores seleccione la opcion 2")
    else:
        print("\nNo se ha realizado previamente ninguna descarga.  Vamos a lanzar la aplicacion completa")
        batch.module1.batchOne()

    if flag_tratamiento != 0:
        # Hacemos el menu de selección de la aplicacion
        # Definimos las opciones
        def opcion_1():
            print("\nHas seleccionado volver a descargar los datos y realizar todo el proceso.")
            batch.module1.batchOne()

        def opcion_2():
            print("\nHas seleccionado comenzar desde el tratamiento de datos")
            if flag_tratamiento > 0:
                batch.module2.batchTwo()
            else:
                print("Faltan datos para poder seleccionar esta acción."
                      "\nEjecuta la acción desde un punto anterior")

        def opcion_3():
            print("\nHas seleccionado comenzar elegiendo clasificador y vectorizador")
            if flag_tratamiento > 1:
                batch.module3_A.batchThree()
            else:
                print("Faltan datos para poder seleccionar esta acción."
                      "\nEjecuta la acción desde un punto anterior")

        def opcion_4():
            print("\nHas seleccionado comenzar arrancando la prevision de la web")
            if flag_tratamiento > 2:
                batch.module4.batchFour(" ", " ")
            else:
                print("Faltan datos para poder seleccionar esta acción."
                      "\nEjecuta la acción desde un punto anterior")

        def opcion_salir():
            print("\nSaliendo del programa.")
            exit()

        # Presentamos el menu en pantalla
        while True:
            # Muestra el menú
            print("\nMenú de Selección:")
            print("1. Descargar los datos")
            print("2. Tratar los datos")
            print("3. Elegir clasificador y vectorizador")
            print("4. Arrancar servicio web - Predicciones")
            print("5. Salir")

            # Solicita la entrada del usuario
            opcion = input("Selecciona una opción (1, 2, 3, 4, 5): ")

            # Realiza acciones según la opción seleccionada
            if opcion == '1':
                opcion_1()
            elif opcion == '2':
                opcion_2()
            elif opcion == '3':
                opcion_3()
            elif opcion == '4':
                opcion_4()
            elif opcion == '5':
                opcion_salir()
            else:
                print("Opción no válida. Por favor, selecciona una opción válida.")


def obtener_datos():
    global ruta_script, ruta_carpeta, file_01
    # obtener ruta si existe dataset DataSet - format TSV
    # file = os.sep + 'SaveDF' + os.sep + 'DSTest_B.tsv'
    file_02 = obtener_ruta_guardado('SaveDF', 'DSTest_B.tsv')
    file_01 = obtener_ruta_guardado('Descargas', 'subtaskB_train.jsonl')

    # Verificar si el archivo existe en la ruta proporcionada
    if os.path.exists(file_01):
        # Obtener la fecha de modificación del archivo
        fecha_modificacion = datetime.datetime.fromtimestamp(os.path.getmtime(file_01))
        # Obtener la fecha de creación del archivo (solo disponible en algunos sistemas)
        fecha_creacion = datetime.datetime.fromtimestamp(os.path.getctime(file_01))

        # Mostrar las fechas
        print(f"El archivo existe.\nFecha de modificación: {fecha_modificacion}")
        if 'fecha_creacion' in locals():
            print(f"Fecha de creación: {fecha_creacion}")
        else:
            print("No se pudo obtener la fecha de creación en este sistema.")
        valores_aceptados = ['S', 'N']
        entrada = 'a'
        while entrada != valores_aceptados:
            entrada = input("\n¿Desea actualizar los datos? S/N:\n")
            if entrada in ['S', 's']:
                while entrada != valores_aceptados:
                    entrada = input("\n¿Desea omitir descarga y tratar los datos existentes? S/N:\n")
                    if entrada in ['S', 's']:
                        batch.module2.batchTwo("modelo", "mi texto")
                    elif entrada in ['N', 'n']:
                        batch.module1.batchOne("modelo", "mi texto")
                    else:
                        print("Seleccione una opción válida")
            elif entrada in ['N', 'n']:
                if os.path.exists(file_02):
                    print("Existe un fichero con los datos tratados.  Los usamos.")
                    batch.module3_B.batchThree("modelo", "mi texto")
                else:
                    print("No existen los datos tratados.  Vamos a volver a generar los datos tratados")
                    batch.module2.batchTwo("modelo", "mi texto")
            else:
                print("Seleccione una opción válida")
    else:
        print("El archivo no existe en la ruta proporcionada.")
        print("Obteniendo datos....")
        batch.module1.batchOne("modelo", "mi texto")


def obtener_datos_web(modelo, texto):
    global ruta_script, ruta_carpeta, file_01
    # obtener ruta si existe dataset DataSet - format TSV
    # file = os.sep + 'SaveDF' + os.sep + 'DSTest_B.tsv'
    file_02 = obtener_ruta_guardado('SaveDF', 'DSTest_B.tsv')
    file_01 = obtener_ruta_guardado('Descargas', 'subtaskB_train.jsonl')

    # Verificar si el archivo existe en la ruta proporcionada
    if os.path.exists(file_01):
        # Obtener la fecha de modificación del archivo
        fecha_modificacion = datetime.datetime.fromtimestamp(os.path.getmtime(file_01))
        # Obtener la fecha de creación del archivo (solo disponible en algunos sistemas)
        fecha_creacion = datetime.datetime.fromtimestamp(os.path.getctime(file_01))

        # Mostrar las fechas
        print(f"El archivo existe.\nFecha de modificación: {fecha_modificacion}")
        if 'fecha_creacion' in locals():
            print(f"Fecha de creación: {fecha_creacion}")
        else:
            print("No se pudo obtener la fecha de creación en este sistema.")
        batch.module1.batchOne()


def guardar_dataset(dfDataSet, archivo):
    print("\nGuardando el fichero...")
    # save DataSet - format TSV
    ruta_carpeta = obtener_ruta_guardado('SaveDF', archivo)
    dfDataSet.to_csv(ruta_carpeta, sep='\t', index=False)
    print("DataFrame save in: " + ruta_carpeta)


def guardar_clf_vct(tipo, fichero, tarea):
    if tipo == 'clf':
        if tarea == 'A':
            ruta_carpeta = obtener_ruta_guardado('SaveCLF', 'clf_A.joblib')
        elif tarea == 'B':
            ruta_carpeta = obtener_ruta_guardado('SaveCLF', 'clf_B.joblib')
        else:
            print("No se puede guardar porque la tarea enviada no es correcta")
            return
    else:
        if tarea == 'A':
            ruta_carpeta = obtener_ruta_guardado('SaveVCT', 'vct_A.joblib')
        elif tarea == 'B':
            ruta_carpeta = obtener_ruta_guardado('SaveVCT', 'vct_B.joblib')
        else:
            print("No se puede guardar porque la tarea enviada no es correcta")
            return
    print("\nGuardando el fichero...")
    dump(fichero, ruta_carpeta)
    print("Clasificador save in: " + ruta_carpeta)


def obtener_ruta_guardado(carpeta, fichero):
    ruta_script = os.path.abspath(__file__)  # Ruta absoluta del script actual
    ruta_carpeta = os.path.dirname(ruta_script)  # Ruta del directorio del script
    ruta_carpeta = (ruta_carpeta[:ruta_carpeta.rfind(os.sep)] + os.sep + carpeta)
    # comprobamos si existe la carpeta
    if not os.path.exists(ruta_carpeta):
        # Si no existe, crear la carpeta
        try:
            os.makedirs(ruta_carpeta)
            print(f"La carpeta '{ruta_carpeta}' ha sido creada.")
        except OSError as error:
            print(f"No se pudo crear la carpeta '{ruta_carpeta}': {error}")
    # else:  # si existe, lo indicamos
    #   print(f"La carpeta '{ruta_carpeta}' ya existe.")
    ruta_completa = ruta_carpeta + os.sep + fichero
    return ruta_completa


def descarga_archivos(archivos):
    for elemento in archivos:
        url, nombre_archivo = elemento  # Desempaquetar la tupla en dos variables
        ruta = obtener_ruta_guardado('Descargas', nombre_archivo)
        gdown.download(url, ruta, fuzzy=True)


def limpia_texto_df(df):
    # Crea una copia del DataFrame para evitar modificar el original
    df_limpio = df.copy()

    # Obtén la cantidad total de filas en el DataFrame
    total_filas = len(df_limpio)

    # Aplica la limpieza solo a las filas donde el campo 'label' es diferente de cero
    # print ("\nAplicar BeautifulSoup")
    # df_limpio['text'] = df_limpio.apply(lambda row: remove_unwanted_tags(row['text']) if row['label'] != 0 else row['text'], axis=1)

    print("\nLimpiar texto")
    pbar = tqdm(total=total_filas)  # Inicializa la barra de progreso
    for index, row in df_limpio.iterrows():
        text = df_limpio.at[index, 'text']
        # Desechamos los textos menores de 20 caracteres
        if len(text) >= 20:
            try:
                # Intentar detectar el idioma si el texto tiene longitud suficiente
                if detect(text) == 'en':
                    df_limpio.at[index, 'text'] = text.replace('\t', '').replace('\n', '').replace('  ', '')
                # Desechamos textos que no sean en ingles
                else:
                    df_limpio.at[index, 'text'] = ''
            except Exception as e:
                # Desechamos textos que generen una excepcion
                df_limpio.at[index, 'text'] = ''
        else:
            df_limpio.at[index, 'text'] = ''
        # Actualiza la barra de progreso
        pbar.update(1)

    # Cierra la barra de progreso al finalizar
    pbar.close()

    # Eliminamos duplicados, filas vacias y filas con algún campo vacio
    print("Eliminando duplicados y filas vacías")
    df_limpio = df_limpio.drop_duplicates()
    df_limpio = df_limpio.dropna(subset=['text'])
    df_limpio = df_limpio.dropna(subset=['label'])
    df_limpio = df_limpio[df_limpio['text'] != '']
    df_limpio = df_limpio[df_limpio['label'] != '']
    print("Tokenizando texto a 50, 150 y 441 tokens")
    # Descargamos datos asolciados al tokenizador Punkt
    nltk.download('punkt')
    # Tokenizamos a 441 tokens - media de los textos sacada de la funcion comentada siguiente:
    #   num=int(df_train_B["text"].map(lambda x: len(x.split(" "))).mean())
    #   print("Numero de caracteres al que reducimos el texto",num)
    df_limpio['tokenized_text'] = df_limpio['text'].apply(tokenize_and_reduce)
    # Tokenizamos a 50 tokens
    df_limpio['tokenized_text_50'] = df_limpio['text'].apply(tokenize_and_reduce_50)
    # Tokenizamos a 150 tokens
    df_limpio['tokenized_text_150'] = df_limpio['text'].apply(tokenize_and_reduce_150)
    return df_limpio


def remove_unwanted_tags(html_content):
    allowed_tags = ["p", "h1", "h2", "h3", "b", "a"]
    # Parsea el contenido HTML utilizando BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    # Busca todas las etiquetas en el contenido HTML
    for tag in soup.find_all():
        if tag.name.lower() not in allowed_tags:
            # Elimina las etiquetas que no están en la lista de etiquetas permitidas
            tag.unwrap()  # Elimina la etiqueta manteniendo su contenido
    # Retorna el contenido limpio sin las etiquetas no permitidas
    return str(soup)


def imprime_estadistica(dfDataSet, name):
    # calcular el número total de instancias
    n_total = len(dfDataSet)

    # Vemos el número de valores diferentes
    n_valores = dfDataSet['label'].value_counts()
    valores_unicos = dfDataSet['label'].unique()

    # dividir instancias humanas y generadas
    dfHuman = dfDataSet[dfDataSet['label'] == 0]
    dfChatGPT = dfDataSet[dfDataSet['label'] == 1]
    dfCohere = dfDataSet[dfDataSet['label'] == 2]
    dfDavinci = dfDataSet[dfDataSet['label'] == 3]
    dfBloomz = dfDataSet[dfDataSet['label'] == 4]
    dfDolly = dfDataSet[dfDataSet['label'] == 5]

    # número de instancias humanas y generadas
    n_humano = len(dfHuman)
    n_chatGPT = len(dfChatGPT)
    n_cohere = len(dfCohere)
    n_davinci = len(dfDavinci)
    n_bloomz = len(dfBloomz)
    n_dolly = len(dfDolly)

    # longitud media de caracteres para instancias humanas y generadas
    long_media_humano = dfHuman['text'].str.len().mean()
    long_media_chatGPT = dfChatGPT['text'].str.len().mean()
    long_media_cohere = dfCohere['text'].str.len().mean()
    long_media_davinci = dfDavinci['text'].str.len().mean()
    long_media_bloomz = dfBloomz['text'].str.len().mean()
    long_media_dolly = dfDolly['text'].str.len().mean()

    # creamos una lista de listas con los datos en funcion del tipo
    if len(n_valores) == 2:
        data = [
            ["Número total de instancias", n_total],
            ["Número de instancias humanas", n_humano],
            ["Número de instancias generadas", n_chatGPT],
            ["Longitud media de instancias humanas", f"{long_media_humano:.2f}"],
            ["Longitud media de instancias generadas", f"{long_media_chatGPT:.2f}"]
        ]
    else:
        data = [
            ["Número total de instancias", n_total],
            ["Número de instancias humanas", n_humano],
            ["Número de instancias Chat GPT", n_chatGPT],
            ["Número de instancias Cohere", n_cohere],
            ["Número de instancias Davinci", n_davinci],
            ["Número de instancias Bloomz", n_bloomz],
            ["Número de instancias Dolly", n_dolly],
            ["Longitud media de instancias humanas", f"{long_media_humano:.2f}"],
            ["Longitud media de instancias Chat GPT", f"{long_media_chatGPT:.2f}"],
            ["Longitud media de instancias Cohere", f"{long_media_cohere:.2f}"],
            ["Longitud media de instancias Davinci", f"{long_media_davinci:.2f}"],
            ["Longitud media de instancias Bloomz", f"{long_media_bloomz:.2f}"],
            ["Longitud media de instancias Dolly", f"{long_media_dolly:.2f}"]
        ]

    # imprimimos los datos en forma de tabla tabulada
    print("\n" + name)
    print(tabulate(data, headers=["Campo", "Valor"], tablefmt="grid"))


def imprime_estadistica_subtarea_B(df_train_B, df_test_B, df_fase_1):
    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    print("Imprimiendo estadistica Subtarea B\n")
    X_train = df_train_B['text']
    y_train = df_train_B['label']
    X_test = df_test_B['text']
    y_test = df_test_B['label']
    X_test_f01 = df_fase_1['text']
    y_test_f01 = df_fase_1['label']
    # print(y_train)

    # Número total de instancias en el dataset original
    n_total = len(df_train_B) + len(df_test_B) + len(df_fase_1)
    # Número de instancias en el conjunto de entrenamiento
    n_train = len(df_train_B)
    # Número de instancias en el conjunto de prueba
    n_test = len(df_test_B)
    # Número de instancias en el conjunto de prueba fase 01
    n_test_f01 = len(df_fase_1)
    # Número de instancias humanas en el conjunto de entrenamiento
    n_human_train = sum(y_train == 0)
    # Número de instancias generadas ChatGPT en el conjunto de entrenamiento
    n_cGPT_train = sum(y_train == 1)
    # Número de instancias generadas cohere en el conjunto de entrenamiento
    n_cohere_train = sum(y_train == 2)
    # Número de instancias generadas davinci en el conjunto de entrenamiento
    n_davinci_train = sum(y_train == 3)
    # Número de instancias generadas bloomz en el conjunto de entrenamiento
    n_bloomz_train = sum(y_train == 4)
    # Número de instancias generadas dolly en el conjunto de entrenamiento
    n_dolly_train = sum(y_train == 5)
    # Número de instancias humanas en el conjunto de prueba
    n_human_test = sum(y_test == 0)
    # Número de instancias generadas ChatGPT en el conjunto de prueba
    n_cGPT_test = sum(y_test == 1)
    # Número de instancias generadas cohere en el conjunto de prueba
    n_cohere_test = sum(y_test == 2)
    # Número de instancias generadas davinci en el conjunto de prueba
    n_davinci_test = sum(y_test == 3)
    # Número de instancias generadas bloomz en el conjunto de prueba
    n_bloomz_test = sum(y_test == 4)
    # Número de instancias generadas dolly en el conjunto de prueba
    n_dolly_test = sum(y_test == 5)
    # Número de instancias humanas en el conjunto de prueba
    n_human_test_f01 = sum(y_test_f01 == 0)
    # Número de instancias generadas en el conjunto de prueba
    n_generated_test_f01 = sum(y_test_f01 == 1)

    # Creamos una lista de listas con los datos
    data = [
        ["Número total de instancias", n_total],
        ["Número de instancias del training", n_train],
        ["Número de instancias del test", n_test],
        ["Número de instancias del test fase_01", n_test_f01],
        ["Número de instancias humanas en el training", n_human_train],
        ["Número de instancias ChatGPT en el training", n_cGPT_train],
        ["Número de instancias Cohere en el training", n_cohere_train],
        ["Número de instancias Davinci en el training", n_davinci_train],
        ["Número de instancias Bloomz en el training", n_bloomz_train],
        ["Número de instancias Dolly en el training", n_dolly_train],
        ["Número de instancias humanas en el test", n_human_test],
        ["Número de instancias ChatGPT en el test", n_cGPT_test],
        ["Número de instancias Cohere en el test", n_cohere_test],
        ["Número de instancias Davinci en el test", n_davinci_test],
        ["Número de instancias Bloomz en el test", n_bloomz_test],
        ["Número de instancias Dolly en el test", n_dolly_test],
        ["Número de instancias humanas en el test fase_01", n_human_test_f01],
        ["Número de instancias generadas en el test fase_01", n_generated_test_f01]
    ]
    # Imprimimos los datos en forma de tabla tabulada
    print(tabulate(data, headers=["Descripción", "Valor"], tablefmt="grid"))


def imprime_estadistica_subtarea_A(df_train_A, df_test_A, df_fase_1):
    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    print("Imprimiendo estadistica Subtarea A\n")
    X_train = df_train_A['text']
    y_train = df_train_A['label']
    X_test = df_test_A['text']
    y_test = df_test_A['label']
    X_test_f01 = df_fase_1['text']
    y_test_f01 = df_fase_1['label']
    # print(y_train)

    # Número total de instancias en el dataset original
    n_total = len(df_train_A) + len(df_test_A) + len(df_fase_1)
    # Número de instancias en el conjunto de entrenamiento
    n_train = len(df_train_A)
    # Número de instancias en el conjunto de prueba
    n_test = len(df_test_A)
    # Número de instancias en el conjunto de prueba fase 01
    n_test_f01 = len(df_fase_1)
    # Número de instancias humanas en el conjunto de entrenamiento
    n_human_train = sum(y_train == 0)
    # Número de instancias generadas en el conjunto de entrenamiento
    n_generated_train = sum(y_train == 1)
    # Número de instancias humanas en el conjunto de prueba
    n_human_test = sum(y_test == 0)
    # Número de instancias generadas en el conjunto de prueba
    n_generated_test = sum(y_test == 1)
    # Número de instancias humanas en el conjunto de prueba
    n_human_test_f01 = sum(y_test_f01 == 0)
    # Número de instancias generadas en el conjunto de prueba
    n_generated_test_f01 = sum(y_test_f01 == 1)

    # Creamos una lista de listas con los datos
    data = [
        ["Número total de instancias", n_total],
        ["Número de instancias del training", n_train],
        ["Número de instancias del test", n_test],
        ["Número de instancias del test fase_01", n_test_f01],
        ["Número de instancias humanas en el training", n_human_train],
        ["Número de instancias generadas en el training", n_generated_train],
        ["Número de instancias humanas en el test", n_human_test],
        ["Número de instancias generadas en el test", n_generated_test],
        ["Número de instancias humanas en el test fase_01", n_human_test_f01],
        ["Número de instancias generadas en el test fase_01", n_generated_test_f01]
    ]
    # Imprimimos los datos en forma de tabla tabulada
    print(tabulate(data, headers=["Descripción", "Valor"], tablefmt="grid"))


def balacearDF(dfDataSet):
    # dividir instancias humanas y generadas
    dfHuman = dfDataSet[dfDataSet['label'] == 0]
    dfIA = dfDataSet[dfDataSet['label'] != 0]

    # número de instancias humanas y generadas
    n_humano = len(dfHuman)
    n_generadas = len(dfIA)

    # balanceamos el numero de resultados
    if n_humano > n_generadas:
        # elegimos de forma random los indices a borrar
        indices_to_remove = np.random.choice(dfHuman.index, (n_humano - n_generadas), replace=False)
        # ajustamos el dataframe al tamaño adecuado
        dfHuman = dfHuman.drop(indices_to_remove)
    if n_humano < n_generadas:
        # elegimos de forma random los indices a borrar
        indices_to_remove = np.random.choice(dfIA.index, (n_generadas - n_humano), replace=False)
        # ajustamos el dataframe al tamaño adecuado
        dfIA = dfIA.drop(indices_to_remove)

    # construimos el dataset final
    dfDataSet_final = pd.concat([dfHuman, dfIA], ignore_index=True)

    return dfDataSet_final
