import sys

from joblib import load
import batch.functions
from batch import functions


# batch 4 - módulo que usamos para cargar clasificador y vectorizador y realizar predicciones
# el vectorizador se utiliza para convertir el texto en una representación numérica antes de alimentarlo al modelo de clasificación
def batchFour(modelo, texto):
    print("\n############ Ejecutando Batch 4: Carga clasificador y realizar predicciones - Uso WEB#############")
    #mi_app = web.app_dash_2.configurar_aplicacion()
    #web.app_dash_2.arrancar_servidor(mi_app)

    if (modelo != " " and texto != " "):
        # limpiamos el texto
        texto = batch.functions.limpia_texto_simple(texto)
        if texto != " ":
            if modelo == 'subtareaA':
                print(modelo, " : \n", texto + "\n")
                # cargamos el clasificador desde el archivo
                loaded_classifier_A = load(functions.obtener_ruta_guardado('SaveCLF', 'clf_A.joblib'))
                print("Clasificador A cargado")
                # cargar el vectorizador desde un archivo
                loaded_vectorizador_A = load(functions.obtener_ruta_guardado('SaveVCT', 'vct_A.joblib'))
                print("Vectorizador A cargado")
                texto_vectorizado = loaded_vectorizador_A.transform([texto])
                # Calcular % de probabilidad de ser de uno u otro tipo
                y_pred_probs = loaded_classifier_A.predict_proba(texto_vectorizado.reshape(1, -1))

                probabilidades = ""
                for class_label, prob in enumerate(y_pred_probs[0]):
                    print(f"Probabilidad para clase {class_label}: {prob:.2%}")
                    probabilidades += f"Probabilidad para clase {class_label}: {prob:.2%}\n"
                # lanzamos la prediccion
                y_pred = loaded_classifier_A.predict(texto_vectorizado)
                if y_pred == 0:
                    print('El texto introducido ha sido escrito por un humano')
                    return 'El texto introducido ha sido escrito por un humano', probabilidades
                else:
                    print('El texto introducio ha sido generado por IA')
                    return 'El texto introducido ha sido escrito por un humano', probabilidades
            else:
                # cargamos el clasificador desde el archivo
                loaded_classifier_B = load(functions.obtener_ruta_guardado('SaveCLF', 'clf_B.joblib'))
                print("Clasificador B cargado")
                # cargar el vectorizador desde un archivo
                loaded_vectorizador_B = load(functions.obtener_ruta_guardado('SaveVCT', 'vct_B.joblib'))
                print("Vectorizador B cargado")
                texto_vectorizado = loaded_vectorizador_B.transform([texto])
                # Calcular % de probabilidad de ser de uno u otro tipo

                probabilidades = ''
                y_pred_probs = loaded_classifier_B.predict_proba(texto_vectorizado.reshape(1, -1))
                for class_label, prob in enumerate(y_pred_probs[0]):
                    print(f"Probabilidad para clase {class_label}: {prob:.2%}")
                    probabilidades += f"Probabilidad para clase {class_label}: {prob:.2%}\n"
                # lanzamos la prediccion
                y_pred = loaded_classifier_B.predict(texto_vectorizado)
                if y_pred == 0:
                    return 'El texto introducido ha sido escrito por un humano', probabilidades
                elif y_pred == 1:
                    return 'El texto introducido ha sido generado por ChatGPT', probabilidades
                elif y_pred == 2:
                    return 'El texto introducido ha sido generado por Cohere', probabilidades
                elif y_pred == 3:
                    return 'El texto introducido ha sido generado por Davinci', probabilidades
                elif y_pred == 4:
                    return 'El texto introducido ha sido generado por Bloomz', probabilidades
                else:
                    return 'El texto introducido ha sido generado por Dolly', probabilidades
        else:
            print('El texto no cumple las caracteristícas necesararias.\nIntroduzca otro texto')
            return 'El texto no cumple las caracteristícas necesararias.\nIntroduzca otro texto', ""
    else:
        print('Esperando a recibir petición web')
