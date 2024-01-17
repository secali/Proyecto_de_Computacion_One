from flask import *
import sys
sys.path.append("../batch") # Añade la ruta de la carpeta batch al sys.path
from batch import functions

app = Flask(__name__)


# código Flask que crea la app web

@app.route('/')
def index():
    # Renderiza el archivo de plantilla index.html
    return render_template('index.html')


# Ruta que recibe datos del formulario a través de 'POST'
@app.route('/resultado', methods=['POST'])
def resultado(resultado=None):
    # Obtiene los datos del formulario
    modelo = request.form['modelo']
    texto = request.form['texto']
    # Aquí puedes usar el modelo de deeplearning que quieras para analizar el texto
    # Por ejemplo, si usas la biblioteca transformers, puedes hacer algo así:
    # from transformers import pipeline
    # nlp = pipeline(modelo)
    # resultado = nlp(texto)
    # Devuelve el resultado del análisis en el archivo de plantilla resultado.html

    functions.obtener_datos_Web()


    return render_template("resultado.html")
