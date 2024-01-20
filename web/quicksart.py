import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State

# Inicializa la aplicación Dash con suppress_callback_exceptions=True
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define el diseño de la aplicación
index_layout = html.Div([
    html.Link(rel='stylesheet', href='/static/style.css'),
    html.H1('Web de deeplearning'),
    html.P('Selecciona un modelo de deeplearning y luego inserta un texto para analizarlo.'),
    html.Div([
        html.Label('Modelo:'),
        dcc.Dropdown(
            id='modelo',
            options=[
                {'label': 'BERT', 'value': 'bert'},
                {'label': 'GPT-3', 'value': 'gpt-3'},
                {'label': 'DistilBERT', 'value': 'distilbert'},
                {'label': 'Transformer', 'value': 'transformer'}
            ],
            value='bert'
        ),
        html.Br(),
        html.Div([
            html.Label('Texto:'),
            html.Br(),
            dcc.Textarea(
                id='texto',
                rows=10,
                cols=50
            ),
        ]),
        html.Br(),
        html.Button("Analizar", id='analizar-button', n_clicks=0),
    ], id='formulario'),

    dcc.Store(id='resultado-store', data='Presiona el botón "Analizar" para ver el saludo'),
    html.Br(),
    html.Div([
        html.Label(id='resultado')
    ]),
])

# Define la lógica de la aplicación
@app.callback(
    [Output('resultado-store', 'data'),
     Output('resultado', 'children')],
    [Input('analizar-button', 'n_clicks')],
    [State('modelo', 'value'),
     State('texto', 'value')]
)
def analizar_texto(n_clicks, modelo, texto):
    if n_clicks is not None and n_clicks > 0:
        # Realiza aquí la lógica de análisis según el modelo y el texto
        resultado = '¡HOLA! Resultado del análisis aquí.'
        return resultado, resultado
    else:
        return 'Presiona el botón "Analizar" para ver el saludo', 'Presiona el botón "Analizar" para ver el saludo'

# Establece el diseño de la aplicación
app.layout = index_layout

# Corre la aplicación en el servidor local
if __name__ == '__main__':
    app.run_server(debug=True)
