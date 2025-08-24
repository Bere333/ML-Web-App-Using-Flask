#from utils import db_connect
import joblib
from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import numpy as np


#engine = db_connect()

app = Flask(__name__)
model = joblib.load('../models/star_prediction_model.pkl')




# Cargar tus datos reales
def cargar_datos_estrellas():
    df = pd.read_csv('../data/processed/data_stars_processed.csv')
    
    return df

# Mapeo de tipos de estrellas
star_type_mapping = {
    0: {'nombre': 'Enana Roja M', 'clase_espectral': 'M', 'color_hex': '#ff6b6b'},
    1: {'nombre': 'Gigante Azul B', 'clase_espectral': 'B', 'color_hex': '#4fc3f7'},
    2: {'nombre': 'Estrella Blanca A', 'clase_espectral': 'A', 'color_hex': '#ffffff'},
    3: {'nombre': 'Subgigante F', 'clase_espectral': 'F', 'color_hex': '#f8f9fa'},
    4: {'nombre': 'Estrella Amarilla G', 'clase_espectral': 'G', 'color_hex': '#ffd54f'},
    5: {'nombre': 'Supergigante O', 'clase_espectral': 'O', 'color_hex': '#64b5f6'}
}

# Información completa de tipos de estrellas
star_types_info = {
    'M': {
        'nombre': 'Enana Roja Tipo M', 
        'color_hex': '#ff6b6b', 
        'descripcion': 'Estrellas rojas pequeñas y frías, las más comunes en la galaxia',
        'color_estrella': 'Rojo',
        'clase_espectral': 'M',
        'temperatura_min': 2400,
        'temperatura_max': 3700,
        'luminosidad_min': 0.0001,
        'luminosidad_max': 0.08,
        'radio_min': 0.1,
        'radio_max': 0.7,
        'magnitud_min': 9,
        'magnitud_max': 20
    },
    'K': {
        'nombre': 'Estrella Naranja Tipo K', 
        'color_hex': '#ffa726', 
        'descripcion': 'Estrellas naranjas de tamaño mediano',
        'color_estrella': 'Naranja',
        'clase_espectral': 'K',
        'temperatura_min': 3700,
        'temperatura_max': 5200,
        'luminosidad_min': 0.08,
        'luminosidad_max': 0.6,
        'radio_min': 0.7,
        'radio_max': 0.96,
        'magnitud_min': 6,
        'magnitud_max': 9
    },
    'G': {
        'nombre': 'Estrella Amarilla Tipo G', 
        'color_hex': '#ffd54f', 
        'descripcion': 'Estrellas amarillas como nuestro Sol',
        'color_estrella': 'Amarillo',
        'clase_espectral': 'G',
        'temperatura_min': 5200,
        'temperatura_max': 6000,
        'luminosidad_min': 0.6,
        'luminosidad_max': 1.5,
        'radio_min': 0.96,
        'radio_max': 1.15,
        'magnitud_min': 4,
        'magnitud_max': 6
    },
    'F': {
        'nombre': 'Estrella Blanco-Amarillenta Tipo F', 
        'color_hex': '#f8f9fa', 
        'descripcion': 'Estrellas blanco-amarillentas más calientes que el Sol',
        'color_estrella': 'Blanco-Amarillento',
        'clase_espectral': 'F',
        'temperatura_min': 6000,
        'temperatura_max': 7500,
        'luminosidad_min': 1.5,
        'luminosidad_max': 5,
        'radio_min': 1.15,
        'radio_max': 1.4,
        'magnitud_min': 2,
        'magnitud_max': 4
    },
    'A': {
        'nombre': 'Estrella Blanca Tipo A', 
        'color_hex': '#e3f2fd', 
        'descripcion': 'Estrellas blancas como Sirio',
        'color_estrella': 'Blanco',
        'clase_espectral': 'A',
        'temperatura_min': 7500,
        'temperatura_max': 10000,
        'luminosidad_min': 5,
        'luminosidad_max': 25,
        'radio_min': 1.4,
        'radio_max': 1.8,
        'magnitud_min': -1,
        'magnitud_max': 2
    },
    'B': {
        'nombre': 'Estrella Azul-Blanca Tipo B', 
        'color_hex': '#90caf9', 
        'descripcion': 'Estrellas azul-blancas muy calientes',
        'color_estrella': 'Azul-Blanca',
        'clase_espectral': 'B',
        'temperatura_min': 10000,
        'temperatura_max': 30000,
        'luminosidad_min': 25,
        'luminosidad_max': 30000,
        'radio_min': 1.8,
        'radio_max': 6.6,
        'magnitud_min': -5,
        'magnitud_max': -1
    },
    'O': {
        'nombre': 'Estrella Azul Tipo O', 
        'color_hex': '#64b5f6', 
        'descripcion': 'Estrellas azules extremadamente calientes y masivas',
        'color_estrella': 'Azul',
        'clase_espectral': 'O',
        'temperatura_min': 30000,
        'temperatura_max': 50000,
        'luminosidad_min': 30000,
        'luminosidad_max': 1000000,
        'radio_min': 6.6,
        'radio_max': 10.0,
        'magnitud_min': -11,
        'magnitud_max': -5
    }
}

@app.route('/')
def index():
    # Pasar datos al template usando render_template
    return render_template('index.html', 
                         star_types_info=star_types_info,
                         default_values={
                             'temperatura': 5800,
                             'luminosidad': 1.0,
                             'radio': 1.0,
                             'magnitud_absoluta': 4.83, 
                             'color_estrella': 'Amarillo',
                             'clase_espectral': 'G'
                         },
                         column_names={
                             'temperatura': 'Temperature (K)',
                             'luminosidad': 'Luminosity(L/Lo)',
                             'radio': 'Radius(R/Ro)',
                             'magnitud_absoluta': 'Absolute magnitude(Mv)',
                             'color_estrella': 'Star color',
                             'clase_espectral': 'Spectral Class'
                         })

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datau = pd.read_csv('../data/processed/data_stars_processed.csv')
        data = datau.to_json(orient='records')
        
        # Validar datos de entrada
        temperatura = float(data['temperatura'])
        luminosidad = float(data['luminosidad'])
        radio = float(data['radio'])
        magnitud_absoluta = float(data['magnitud_absoluta'])
        color_estrella = float(data['luminosidad']),
        clase_espectral = float(data['luminosidad'])
        
        # Crear array de características en el orden correcto
        features = np.array([[temperatura, luminosidad, radio, magnitud_absoluta, color_estrella, clase_espectral]])
        
        # Hacer predicción
        prediccion = model.predict(features)[0]
        
        # Obtener información del tipo de estrella
        info_estrella = star_types_info.get(prediccion, {
            'nombre': 'Desconocido', 
            'color_hex': '#cccccc', 
            'descripcion': 'Tipo no identificado',
            'color_estrella': 'Desconocido',
            'clase_espectral': '?'
        })
        
        return jsonify({
            'exito': True,
            'prediccion': prediccion,
            'nombre_estrella': info_estrella['nombre'],
            'color_hex': info_estrella['color_hex'],
            'descripcion': info_estrella['descripcion'],
            'color_estrella': info_estrella['color_estrella'],
            'clase_espectral': info_estrella['clase_espectral'],
            'rango_temperatura': f"{info_estrella['temperatura_min']:,} - {info_estrella['temperatura_max']:,} K".replace(',', '.'),
            'rango_magnitud': f"{info_estrella['magnitud_min']} a {info_estrella['magnitud_max']} Mv"
        })
        
    except Exception as e:
        return jsonify({
            'exito': False,
            'error': str(e)
        })

@app.route('/ejemplos')
def obtener_ejemplos():
    # Proporcionar ejemplos de diferentes tipos de estrellas
    ejemplos = []
    for tipo_estrella, info in star_types_info.items():
        ejemplos.append({
            'tipo': tipo_estrella,
            'nombre': info['nombre'],
            'color_hex': info['color_hex'],
            'color_estrella': info['color_estrella'],
            'clase_espectral': info['clase_espectral'],
            'temperatura_tipica': (info['temperatura_min'] + info['temperatura_max']) // 2,
            'luminosidad_tipica': (info['luminosidad_min'] + info['luminosidad_max']) / 2,
            'radio_tipico': (info['radio_min'] + info['radio_max']) / 2,
            'magnitud_tipica': (info['magnitud_min'] + info['magnitud_max']) / 2
        })
    
    return jsonify(ejemplos)

@app.route('/datos_originales')
def datos_originales():
    # Devolver algunos datos originales del dataframe
    df = cargar_datos_estrellas()
    datos_sample = df.head(10).to_dict('records')
    return jsonify(datos_sample)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


