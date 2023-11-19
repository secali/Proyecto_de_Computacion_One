import batch.module1
import batch.module2
import os
import datetime
from joblib import dump

def obtener_datos():
    global ruta_script, ruta_carpeta, file
    # obtener ruta si existe dataset DataSet - format TSV
    obtener_ruta_existe()

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


def obtener_ruta_existe():
    global ruta_script, ruta_carpeta, file
    ruta_script = os.path.abspath(__file__)  # Ruta absoluta del script actual
    ruta_carpeta = os.path.dirname(ruta_script)  # Ruta del directorio del script
    # ruta_carpeta = ruta_carpeta[:ruta_carpeta.rfind(os.sep)] + os.sep + 'SaveDF'
    file = os.sep + 'SaveDF' + os.sep + 'DataFrame.tsv'


def guardar_dataset(dfDataSet):
    print("\nGuardando el fichero...")
    # save DataSet - format TSV
    file, ruta_carpeta = obtener_ruta_guardado()
    # print(ruta_carpeta + file)
    # comprobamos si existe la carpeta
    if not os.path.exists(ruta_carpeta):
        # Si no existe, crear la carpeta
        try:
            os.makedirs(ruta_carpeta)
            print(f"La carpeta '{ruta_carpeta}' ha sido creada.")
        except OSError as error:
            print(f"No se pudo crear la carpeta '{ruta_carpeta}': {error}")
    else:  # si existe, lo indicamos
        print(f"La carpeta '{ruta_carpeta}' ya existe.")
    # guardamos el dataset en csv tabulado
    dfDataSet.to_csv(ruta_carpeta + file, sep='\t', index=False)
    print("Data frame save in: " + ruta_carpeta + file)

def guardar_clf(clf):
    print("\nGuardando el fichero...")
    # save DataSet - format TSV
    file, ruta_carpeta = obtener_ruta_guardado_clf()
    # print(ruta_carpeta + file)
    # comprobamos si existe la carpeta
    if not os.path.exists(ruta_carpeta):
        # Si no existe, crear la carpeta
        try:
            os.makedirs(ruta_carpeta)
            print(f"La carpeta '{ruta_carpeta}' ha sido creada.")
        except OSError as error:
            print(f"No se pudo crear la carpeta '{ruta_carpeta}': {error}")
    else:  # si existe, lo indicamos
        print(f"La carpeta '{ruta_carpeta}' ya existe.")
    # guardamos clasificador en formato .joblib
    dump(clf,ruta_carpeta+file)
    print("Clasificador save in: " + ruta_carpeta + file)

def guardar_vct(vct):
    print("\nGuardando el fichero...")
    # save DataSet - format TSV
    file, ruta_carpeta = obtener_ruta_guardado_vct()
    # print(ruta_carpeta + file)
    # comprobamos si existe la carpeta
    if not os.path.exists(ruta_carpeta):
        # Si no existe, crear la carpeta
        try:
            os.makedirs(ruta_carpeta)
            print(f"La carpeta '{ruta_carpeta}' ha sido creada.")
        except OSError as error:
            print(f"No se pudo crear la carpeta '{ruta_carpeta}': {error}")
    else:  # si existe, lo indicamos
        print(f"La carpeta '{ruta_carpeta}' ya existe.")
    # guardamos vectorizador en formato .joblib
    dump(vct,ruta_carpeta+file)
    print("Vectorizador save in: " + ruta_carpeta + file)

def obtener_ruta_guardado():
    ruta_script = os.path.abspath(__file__)  # Ruta absoluta del script actual
    ruta_carpeta = os.path.dirname(ruta_script)  # Ruta del directorio del script
    ruta_carpeta = ruta_carpeta[:ruta_carpeta.rfind(os.sep)] + os.sep + 'SaveDF'
    file = os.sep + 'DataFrame.tsv'
    return file, ruta_carpeta

def obtener_ruta_guardado_clf():
    ruta_script = os.path.abspath(__file__)  # Ruta absoluta del script actual
    ruta_carpeta = os.path.dirname(ruta_script)  # Ruta del directorio del script
    ruta_carpeta = ruta_carpeta[:ruta_carpeta.rfind(os.sep)] + os.sep + 'SaveCLF'
    file = os.sep + 'clf.joblib'
    return file, ruta_carpeta
def obtener_ruta_guardado_vct():
    ruta_script = os.path.abspath(__file__)  # Ruta absoluta del script actual
    ruta_carpeta = os.path.dirname(ruta_script)  # Ruta del directorio del script
    ruta_carpeta = ruta_carpeta[:ruta_carpeta.rfind(os.sep)] + os.sep + 'SaveVCT'
    file = os.sep + 'vct.joblib'
    return file, ruta_carpeta

def obtener_ruta():
    ruta_script = os.path.abspath(__file__)  # Ruta absoluta del script actual
    ruta_carpeta = os.path.dirname(ruta_script)  # Ruta del directorio del script
    ruta_carpeta = ruta_carpeta[:ruta_carpeta.rfind(os.sep)] + os.sep + 'SaveDF'
    file = ruta_carpeta + os.sep + 'DataFrame.tsv'
    return file