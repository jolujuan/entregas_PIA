from flask import Flask, request, jsonify
from utilities import load_model, predict_sentiment_with_confidence
from transformers import TextCleaner, StopwordRemover
import os

app = Flask(__name__)

# Cargar el modelo al iniciar la aplicación Flask
try:
    load_model()
except FileNotFoundError as e:
    print(f"ADVERTENCIA: {e}")
    print("La API se ha inicializado, pero el endpoint /predict_sentiment falla.")
    print("Asegúrate de ejecutar 'python models/train.py' para crear el modelo.")

@app.route('/')
def home():
    # Endpoint raíz simple para verificar que la API está viva
    return jsonify({"status": "API de Sentimientos en funcionamiento. "
                              "Usa el endpoint /predict_sentiment para predecir."})

@app.route('/predict_sentiment', methods=['POST'])
def predict():
    """
    Endpoint de la API para predicción de sentimientos.
    Espera un JSON con la clave "text".
    El valor puede ser un string o una lista de strings.
    
    Ejemplo de JSON (un solo texto):
    {
        "text": "This is a great product!"
    }
    
    Ejemplo de JSON (múltiples textos):
    {
        "text": ["This is great!", "This is terrible."]
    }
    
    Response (un solo texto):
    {
        "text": "This is a great product!",
        "prediction": 1,
        "label": "positivo",
        "confidence": 0.9996
    }
    
    Response (múltiples textos):
    [
        {
            "text": "This is great!",
            "prediction": 1,
            "label": "positivo",
            "confidence": 0.9950
        },
        {
            "text": "This is terrible.",
            "prediction": 0,
            "label": "negativo",
            "confidence": 0.9977
        }
    ]
    """
    if not request.is_json:
        return jsonify({"error": "La solicitud debe ser de tipo JSON"}), 400
        
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "JSON debe contener la clave 'text'"}), 400
        
    text_input = data['text']
    
    try:
        # Realizar predicciones con confianza
        results = predict_sentiment_with_confidence(text_input)
        
        # Si la entrada fue un solo string, devolver un solo objeto
        if isinstance(text_input, str):
            pred, confianza, label = results[0]
            response = {
                "text": text_input,
                "prediction": pred,
                "label": label.lower(),
                "confidence": confianza
            }
        # Si la entrada fue una lista, devolver una lista de objetos
        else:
            response = [
                {
                    "text": text_input[i],
                    "prediction": results[i][0],
                    "label": results[i][2].lower(),
                    "confidence": results[i][1]
                } for i in range(len(text_input))
            ]
            
        return jsonify(response)

    except FileNotFoundError:
        return jsonify({"error": "Modelo no encontrado. "
                                 "El administrador debe entrenar el modelo primero."}), 500
    except Exception as e:
        # Captura genérica para otros errores de predicción
        return jsonify({"error": f"Error durante la predicción: {str(e)}"}), 500

if __name__ == '__main__':
    # Obtener el puerto de la variable de entorno o usar 5000 por defecto
    port = int(os.environ.get("PORT", 5000))
    # 'debug=True' es útil para desarrollo, pero desactívalo en producción
    # 'host=0.0.0.0' es necesario para que sea accesible desde Docker
    app.run(debug=True, host='0.0.0.0', port=port)