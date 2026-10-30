import joblib
import os
from transformers import TextCleaner, StopwordRemover

# --- Lógica de Carga y Predicción ---

MODEL_PATH = 'models/pipeline_sentimientos.joblib'
loaded_pipeline = None

def load_model():
    """Carga el pipeline desde el disco a una variable global."""
    global loaded_pipeline
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encuentra el modelo en {MODEL_PATH}. "
                                "Ejecuta `python models/train.py` primero.")
    
    print(f"Cargando modelo desde {MODEL_PATH}...")
    loaded_pipeline = joblib.load(MODEL_PATH)
    print("¡Modelo cargado exitosamente!")

def predict_sentiment(text_list):
    """
    Realiza predicciones de sentimiento (0 o 1) sobre una lista de textos.
    """
    if loaded_pipeline is None:
        load_model()
        
    if not isinstance(text_list, list):
        text_list = [text_list]
        
    predictions = loaded_pipeline.predict(text_list)
    # Convertir de numpy.int64 a int nativo de Python para serialización JSON
    return [int(pred) for pred in predictions]

def predict_sentiment_with_confidence(text_list):
    """
    Devuelve predicción y confianza de la clase predicha.
    
    Returns:
        list of tuples: [(predicción, confianza, label), ...]
        - predicción: 0 (Negativo) o 1 (Positivo)
        - confianza: probabilidad de la clase predicha (0.0 a 1.0)
        - label: nombre legible ("Negativo" o "Positivo")
    """
    if loaded_pipeline is None:
        load_model()

    if not isinstance(text_list, list):
        text_list = [text_list]
    
    # Obtener predicciones
    predictions = loaded_pipeline.predict(text_list)
    
    # Obtener probabilidades: [prob_clase_0, prob_clase_1]
    probabilities = loaded_pipeline.predict_proba(text_list)
    
    results = []
    for pred, probs in zip(predictions, probabilities):
        # La confianza es la probabilidad de la clase predicha
        confianza = probs[int(pred)]
        label = "Positivo" if pred == 1 else "Negativo"
        results.append((int(pred), float(confianza), label))
    
    return results

def predict_sentiment_proba(text_list):
    """
    Devuelve las probabilidades de ambas clases para una lista de textos.
    
    Returns:
        list of tuples: [(prob_negativo, prob_positivo), ...]
    """
    if loaded_pipeline is None:
        load_model()

    if not isinstance(text_list, list):
        text_list = [text_list]
        
    # predict_proba devuelve [prob_clase_0, prob_clase_1]
    probabilities = loaded_pipeline.predict_proba(text_list)
    
    # Devolvemos ambas probabilidades
    return [(float(prob[0]), float(prob[1])) for prob in probabilities]

# Comprobación al arrancar
if __name__ == "__main__":
    print("Ejecutando prueba de utilidades...\n")
    load_model()
    
    test_phrases = [
        "What a great product! Love it!",
        "Terrible service, horrible day... #fail",
        "This is just OK, not great, not bad."
    ]
    
    results = predict_sentiment_with_confidence(test_phrases)
    
    print("--- Resultados de la Prueba ---")
    for i, phrase in enumerate(test_phrases):
        pred, confianza, label = results[i]
        print(f"Frase: '{phrase}'")
        print(f"  -> Predicción: {label} ({pred})")
        print(f"  -> Confianza ({label}): {confianza:.4f}\n")