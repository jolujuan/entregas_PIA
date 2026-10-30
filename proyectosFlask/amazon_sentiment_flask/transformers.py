"""
Módulo compartido con las clases de transformación del pipeline.
Debe estar importado tanto en train.py como en app.py y utilities.py
para que joblib pueda deserializar correctamente el modelo.
"""

import re
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Transformador de pipeline para limpiar texto.
    """
    
    def _limpiar_texto(self, text):
        """Función de limpieza interna."""
        text = str(text)
        sin_saltos = re.sub('\n', ' ', text.lower())
        sin_url = re.sub(r'http\S+', '', sin_saltos)
        sin_nickname = re.sub(r'@\S+', '', sin_url)
        sin_hashtag = re.sub(r'#\S+', '', sin_nickname)
        sin_character = re.sub(r'[^\w\s]', '', sin_hashtag)
        sin_numero = re.sub(r'\d+', '', sin_character)
        sin_tildes = sin_numero.translate(str.maketrans('áéíóúü', 'aeiouu'))
        return sin_tildes.strip()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._limpiar_texto(text) for text in X]


class StopwordRemover(BaseEstimator, TransformerMixin):
    """
    Transformador de pipeline para eliminar stopwords en inglés.
    """
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

    def _eliminar_stopwords(self, text):
        """Función interna de eliminación de stopwords."""
        words = text.split()
        cleaned_text = ' '.join([word for word in words if word not in self.stop_words])
        return cleaned_text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._eliminar_stopwords(text) for text in X]