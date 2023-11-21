import batch.module1_ant
import batch.module1
import batch.module2
import os
import datetime
from joblib import dump
from bs4 import BeautifulSoup

def extraer_texto_iAGenerated(text):
    soup = BeautifulSoup(text, 'html.parser')
    tags_permitidas = ["p", "h1", "h2", "h3", "b", "a"]
    texto_extraido = ' '.join([tag.get_text() for tag in soup.find_all(tags_permitidas)])
    return texto_extraido

def obtener_datos():
    global ruta_script, ruta_carpeta, file
    # obtener ruta si existe dataset DataSet - format TSV
    file = os.sep + 'SaveDF' + os.sep + 'DataFrame.tsv'

    # Verificar si el archivo existe en la ruta proporcionada
    if os.path.exists(file):
        # Obtener la fecha de modificación del archivo
        fecha_modificacion = datetime.datetime.fromtimestamp(os.path.getmtime(file))
        # Obtener la fecha de creación del archivo (solo disponible en algunos sistemas)
        fecha_creacion = datetime.datetime.fromtimestamp(os.path.getctime(file))

        # Mostrar las fechas
        print(f"El archivo existe.\nFecha de modificación: {fecha_modificacion}")
        if 'fecha_creacion' in locals():
            print(f"Fecha de creación: {fecha_creacion}")
        else:
            print("No se pudo obtener la fecha de creación en este sistema.")
        valores_aceptados= ['S','N']
        entrada = 'a'
        while entrada != valores_aceptados:
            entrada = input("\n¿Desea actualizar los datos? S/N:\n")
            if entrada in ['S','s']:
                print("Obteniendo datos....")
                batch.module1.batchOne()
            elif entrada in ['N', 'n']:
                batch.module2.batchTwo()
            else:
                print ("Seleccione una opción válida")
    else:
        print("El archivo no existe en la ruta proporcionada.")
        print ("Obteniendo datos....")
        batch.module1.batchOne()


def guardar_dataset(dfDataSet, archivo):
    print("\nGuardando el fichero...")
    # save DataSet - format TSV
    ruta_carpeta = obtener_ruta_guardado('SaveDF', archivo)
    dfDataSet.to_csv(ruta_carpeta, sep='\t', index=False)
    print("Data frame save in: " + ruta_carpeta)


def guardar_clf_vct(tipo, fichero):
    if tipo =='clf':
        ruta_carpeta = obtener_ruta_guardado('SaveCLF', 'CLF.joblib')
    else :
        ruta_carpeta = obtener_ruta_guardado('SaveVCT', 'vct.joblib')
    print("\nGuardando el fichero...")
    dump(fichero ,ruta_carpeta)
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
    #else:  # si existe, lo indicamos
     #   print(f"La carpeta '{ruta_carpeta}' ya existe.")
    ruta_completa = ruta_carpeta + os.sep + fichero
    return ruta_completa
