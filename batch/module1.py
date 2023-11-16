# pip install langdetect

import requests
import json
from bs4 import BeautifulSoup
from langdetect import detect
import pandas as pd
import os



# We will store all methods and variables related to ShareGPT data extraction.


def batchOne():
    ENGLISH_TAG = 'en'
    url = "https://google.serper.dev/search"  # serper url
    extractedLinks = []  # links obtained from Serper
    humanGeneratedList = []  # text generated by human
    iAGeneratedList = []  # text generated by IA

    # method that extracts Serper links and visits them to extract Human and AI conversations.

    payload = json.dumps({
        "q": "site:sharegpt.com"
    })
    headers = {
        'X-API-KEY': 'f89a56ab46725993c40a6939284b05fdfe7ecce4',
        'Content-Type': 'application/json'
    }

    # Get request from Serper
    response = requests.request("POST", url, headers=headers, data=payload)
    #print(response.json())

    # Filter to obtain the links
    extractedLinks = [item['link'] for item in response.json()['organic']]
    print(extractedLinks)

    # Visit all links and GET clasify conversation in human or generated

    for thisLink in extractedLinks:
        thisResponse = requests.get(thisLink)
        #print(thisResponse.text)


        soup = BeautifulSoup(thisResponse.text, 'html.parser')
        humanGeneratedList = soup.findAll('p', class_='pb-2 whitespace-prewrap')

        # print(humanGeneratedList)

        soup = BeautifulSoup(thisResponse.text, 'html.parser')
        iAGeneratedList = soup.findAll('div', class_='utils_response__b5jEi')

        # print(AIGeneratedList)

    print('There are: ', len(humanGeneratedList), 'items in humanGeneratedList')
    print('There are: ', len(iAGeneratedList), 'items in AIGeneratedList')

    # Triying to clean html tags after human/AI filter, less than 20 character and
    # not english text

    cleanHumanGeneratedList = []
    typeHumanList = []
    cleanIaGeneratedList = []
    typeIAList = []

    for item in humanGeneratedList:

        soup = BeautifulSoup(item.text, "html.parser")
        text = soup.text

        if len(text) > 20 and detect(text) == ENGLISH_TAG:
            #cleanHumanGeneratedList.append(text)
            cleanHumanGeneratedList.append(text.strip().replace('\t', '').replace('\n', ''))
            print(text.strip().replace('\t', '').replace('\n', ''))
            typeHumanList.append('h') # añadimos etiqueta de humano
        else:
            print('Removed text')

    for item in iAGeneratedList:

        soup = BeautifulSoup(item.text, "html.parser")
        text = soup.text

        if len(text) > 20 and detect(text) == ENGLISH_TAG:
            #cleanIaGeneratedList.append(text)
            cleanIaGeneratedList.append(text.strip().replace('\t', '').replace('\n', ''))
            print(text.strip().replace('\t', '').replace('\n', ''))
            typeIAList.append('g') # añadimos etiqueta de generado
        else:
            print('Removed text')

    # generamos diccionarios con los arrais
    datosHuman ={
        'Text': cleanHumanGeneratedList,
        'Type': typeHumanList
    }
    datosIA ={
        'Text': cleanIaGeneratedList,
        'Type': typeIAList
    }
    # creamos los dataFrame y concatenamos.  Generamos un DataSet completo
    dfHuman = pd.DataFrame(datosHuman)
    dfIA = pd.DataFrame(datosIA)
    dfDataSet = pd.concat([dfHuman,dfIA], ignore_index=True)

    print (dfDataSet)

    # save DataSet - format TSV
    ruta_script = os.path.abspath(__file__)  # Ruta absoluta del script actual
    ruta_carpeta = os.path.dirname(ruta_script)  # Ruta del directorio del script
    ruta_carpeta = ruta_carpeta[:ruta_carpeta.rfind(os.sep)]+os.sep+'SaveDF'
    file = os.sep+'DataFrame.tsv'
    print(ruta_carpeta+file)

    # comprobamos si existe la carpeta
    if not os.path.exists(ruta_carpeta):
        # Si no existe, crear la carpeta
        try:
            os.makedirs(ruta_carpeta)
            print(f"La carpeta '{ruta_carpeta}' ha sido creada.")
        except OSError as error:
            print(f"No se pudo crear la carpeta '{ruta_carpeta}': {error}")
    else: # si existe, lo indicamos
        print(f"La carpeta '{ruta_carpeta}' ya existe.")
    # guardamos el dataset en csv tabulado
    dfDataSet.to_csv(ruta_carpeta+file, sep='\t', index=False)
    print("Data frame save in: "+ruta_carpeta+file)