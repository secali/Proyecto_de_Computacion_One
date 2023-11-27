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

    # definimos las listas que vamos a usar
    ListaHumanosClean = set()
    ListaGeneradosClean = set()

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
    df = df[df['title'].str.contains('ShareGPT conversation')]
    print("Quedan ", len(df), " links oprativos para usar")

    # definimos tags para la IA
    allowed_tags = ["p", "h1", "h2", "h3", "b", "a"]
    # pasamos los link a una lista
    links = df['link'].values.tolist()
    # recorremos los links
    for url in links:
        # Enviamos peticion GET al URL
        response = requests.get(url)
        if response.status_code == 200:  # Verificamos si la pagina se descargo correctamente
            print("Descarga Correcta: ", url)
            # Extraemos el contenido HTML legible de la pagina
            soup = BeautifulSoup(response.text, 'html.parser')
            # Buscamos etiqueta de humano
            ListaHumanos = soup.find_all('p', class_='pb-2 whitespace-prewrap')
            # Buscamos etiqueta de generado
            ListaGenerados = soup.find_all('div', class_='utils_response__b5jEi')
            # Filtramos la lista de generados pasandolos por los tag que nos han indicado
            for elemento in ListaGenerados:
                if elemento.name in allowed_tags:
                    ListaGeneradosClean.add(elemento.getText())
            # Creamos la lista con los textos limpios
            ListaHumanosClean.update(batch.functions.limpia_texto(ListaHumanos))
            ListaGeneradosClean.update(batch.functions.limpia_texto(ListaGenerados))
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

    # creamos los dataFrame y concatenamos.  Generamos un DataSet completo
    dfHuman = pd.DataFrame({'Text':list(ListaHumanosClean)})
    dfIA = pd.DataFrame ({'Text':list(ListaGeneradosClean)})
    # añadimos el tipo en la columna Label
    dfHuman ['Label'] = 'h'
    dfIA['Label'] = 'g'

    dfDataSet = pd.concat([dfHuman, dfIA], ignore_index=True)


    # eliminamos duplicados si existen
    dfDataSet.drop_duplicates()

    # guardamos el DataSet
    batch.functions.guardar_dataset(dfDataSet, 'DataFrame.tsv')

    # simulamos el salto al modulo 2, que continuaria con las operaciones requeridas.
    batch.module2.batchTwo()
