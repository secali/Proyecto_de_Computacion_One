from flask import Flask, render_template, request
from main import runScript

app = Flask(__name__)


@app.route('/')
def index():
    # Renderiza el archivo de plantilla index.html
    return render_template('index.html')


@app.route('/analizar', methods=['POST'])
def analizar(resultado=None):
    # Obtiene los datos del formulario
    modelo = request.form['modelo']
    texto = request.form['texto']
    # Aquí puedes usar el modelo de deeplearning que quieras para analizar el texto
    # Por ejemplo, si usas la biblioteca transformers, puedes hacer algo así:
    # from transformers import pipeline
    # nlp = pipeline(modelo)
    # resultado = nlp(texto)
    # Devuelve el resultado del análisis en el archivo de plantilla resultado.html

    runScript() #LANZA NUESTRO SCRIPT DE LA PRACTICA 1
    return render_template('resultado.html', resultado=resultado)
