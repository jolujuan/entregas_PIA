import os
import joblib
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from transformers import TextCleaner, StopwordRemover

# --- Configuración de Archivos ---

CSV_FILE_NAME_LOCAL = "amazon_sentiment.csv"
DATA_DIR = 'data'
LOCAL_DATA_PATH = os.path.join(DATA_DIR, CSV_FILE_NAME_LOCAL)
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'pipeline_sentimientos.joblib')

# --- Lógica Principal de Entrenamiento ---

def train_and_save_pipeline():
    """
    Función principal para entrenar y guardar el pipeline con SMOTE.
    SMOTE se encarga del desbalance, sin necesidad de sample_weight.
    """
    # 1. Descargar recursos de NLTK
    print("Descargando 'stopwords' de NLTK...")
    try:
        nltk.data.find('corpora/stopwords')
    #except nltk.downloader.DownloadError:
    except LookupError:

        nltk.download('stopwords')
    
    # 2. Cargar datos crudos desde la ruta local estable
    print(f"Cargando datos crudos desde: {LOCAL_DATA_PATH}...")
    
    if not os.path.exists(LOCAL_DATA_PATH):
        print("--- ¡ERROR FATAL DE CARGA! ---")
        print(f"El archivo '{CSV_FILE_NAME_LOCAL}' no se encuentra en la carpeta '{DATA_DIR}'.")
        print("Asegúrese de haberlo descargado manualmente y colocado ahí.")
        return
        
    try:
        df_raw = pd.read_csv(LOCAL_DATA_PATH, encoding='latin-1') 
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return

    # 3. Preparación de datos
    df_raw = df_raw[["review_body", "sentiment"]].rename(columns={"review_body": "text"}).dropna(subset=['text'])

    X_raw = df_raw["text"]
    y_raw = df_raw["sentiment"]
    
    print(f"Distribución de clases antes de split:")
    print(y_raw.value_counts())
    
    # Dividimos los datos crudos
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    print(f"\nDatos divididos: {len(X_train_raw)} para entrenamiento, {len(X_test_raw)} para prueba.")
    print(f"Distribución en entrenamiento:")
    print(y_train_raw.value_counts())

    # 4. Definir el pipeline CON SMOTE
    print("\nDefiniendo el pipeline con SMOTE...")
    pipeline_completo = ImbPipeline([
        ('1_limpieza', TextCleaner()),
        ('2_stopwords', StopwordRemover()),
        ('3_vectorizer', CountVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5)),
        ('4_smote', SMOTE(random_state=42)),
        ('5_clasificador', MultinomialNB())
    ])

    # 5. Entrenar el pipeline
    print("Entrenando el pipeline completo con SMOTE... (Esto puede tardar)")
    
    # FIT SIN sample_weight (SMOTE se encarga del balanceo)
    pipeline_completo.fit(X_train_raw, y_train_raw)
    
    print("¡Pipeline entrenado!")
    
    # Evaluar el pipeline en el set de prueba
    accuracy = pipeline_completo.score(X_test_raw, y_test_raw)
    print(f"Precisión (Accuracy) en el set de prueba: {accuracy:.4f}")

    # 6. Guardar el pipeline
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    joblib.dump(pipeline_completo, MODEL_PATH)
    print(f"¡Pipeline completo guardado en '{MODEL_PATH}'!")

if __name__ == "__main__":
    train_and_save_pipeline()