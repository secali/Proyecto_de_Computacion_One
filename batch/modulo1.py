# import required libraries

import requests
import json
from bs4 import BeautifulSoup
from langdetect import detect
import pandas as pd
import batch.module2
import batch.functions
import http.client


# batch 1 - lo usamos para hacer el crawling y el scraping
def batchOne():
    print("\n############Ejecutando Batch 1: Crawling y Scraping#############\n")
    ENGLISH_TAG = 'en'  # definimos el idioma que vamos a usar
    url = "https://google.serper.dev/search"  # serper url

    # definimos las listas que vamos a usar
    humanGeneratedList = []
    iAGeneratedList = []
    cleanHumanGeneratedList = []
    typeHumanList = []
    cleanIaGeneratedList = []
    typeIAList = []

    print("Descargando links de sharegpt.com")
    # definimos la conexión
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": "site:sharegpt.com",
        "num": 100
    })
    headers = {
        'X-API-KEY': '1ce666e8226e13a920c07399dabfbe95c500d087',
        'Content-Type': 'application/json'
    }
    # creamos conexion de la consulta
    conn.request("POST", "/search", payload, headers)
    # obtenemos la respuesta
    response = conn.getresponse()
    # guardamos la respuesta
    data = response.read()
    # decodificamos la respuesta
    decoded_data = json.loads(data)
    # crear un DataFrame e incluimos la respuesta
    # filtramos contenidos en la clave organic
    df = pd.DataFrame(decoded_data['organic'])

    print("Links descargados totales: ", len(df))

    # eliminamos lo que no sean conversaciones antes de hacer la descarga
    iEliminar = []
    print("Borrando los links que no contienen conversaciones")
    # eliminamos links que no incluyan en titulo  'ShareGPT conversation'
    for index, row in df.iterrows():
        if 'ShareGPT conversation' not in row['title']:
            iEliminar.append(index)
    df = df.drop(iEliminar)
    print("Quedan ", len(df), " links oprativos para usar")

    # hacemos separacion de humano y generado
    print("Descargando los textos de cada link y realizando la separacion entre humanos y generados")
    for index, row in df.iterrows():
        response = requests.get(row['link'])
        soup = BeautifulSoup(response.text, 'html.parser')
        # buscamos etiqueta que identifica humano
        humanGeneratedList.append(soup.find_all('p', class_='pb-2 whitespace-prewrap'))
        # buscamos etiqueta que identifica texto generado
        iAGeneratedList.append(soup.find_all('div', class_='utils_response__b5jEi'))

    # filtramos los textos humanos y eliminamos los que no cumplen las condiciones
    print("Filtrando los textos humanos y eliminando los que no cumplen las condiciones")
    for extractedResponses in humanGeneratedList:
        for item in extractedResponses:
            # parseamos con BeautifulSoup
            soup = BeautifulSoup(item.text, "html.parser")
            text = soup.text # extraemos el texto
            # filtramos y creamos lista filtrada
            if len(text) > 20 and detect(text) == ENGLISH_TAG:
                cleanHumanGeneratedList.append(text.strip().replace('\t', '').replace('\n', ''))
                typeHumanList.append('h')  # añadimos etiqueta de humano

    # filtramos los textos humanos y eliminamos los que no cumplen las condiciones
    print("Filtrando los textos generados y eliminando los que no cumplen las condiciones")
    for extractedResponses in iAGeneratedList:
        print(extractedResponses)
        for item in extractedResponses:
            # parseamos con BeautifulSoup
            soup = BeautifulSoup(item.text, "html.parser")
            text = soup.text
            '''texto_extraido = []
            # Encontramos todos los tags que coinciden con los tipos específicos
            tags_permitidas = ["p", "h1", "h2", "h3", "b", "a"]
            tags_encontrados = soup.find_all(tags_permitidas)

            # Extraemos el texto de los tags encontrados y lo agregamos a la lista de texto extraído
            texto_tags = ' '.join([tag.get_text() for tag in tags_encontrados])
            texto_extraido.append(texto_tags)'''
            # filtramos y creamos lista filtrada
            if len(text) > 20 and detect(text) == ENGLISH_TAG:
                cleanIaGeneratedList.append(text.strip().replace('\t', '').replace('\n', ''))
                typeIAList.append('g')  # añadimos etiqueta de generado

    # generamos diccionarios con los arrays
    datosHuman = {
        'Text': cleanHumanGeneratedList,
        'Type': typeHumanList
    }
    datosIA = {
        'Text': cleanIaGeneratedList,
        'Type': typeIAList
    }
    # creamos los dataFrame y concatenamos.  Generamos un DataSet completo
    dfHuman = pd.DataFrame(datosHuman)
    dfIA = pd.DataFrame(datosIA)
    dfDataSet = pd.concat([dfHuman, dfIA], ignore_index=True)

    # eliminamos duplicados si existen
    dfDataSet.drop_duplicates()

    # guardamos el DataSet
    batch.functions.guardar_dataset(dfDataSet, 'DataFrame.tsv')

    # simulamos el salto al modulo 2, que continuaria con las operaciones requeridas.
    batch.module2.batchTwo()
