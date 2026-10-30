# Configuramos la imagen oficial de Python 3.11.2
FROM python:3.11.2

# Establecer el directorio de trabajo
WORKDIR /app

# Copia el archivo requirements.txt primero para cachear la capa de dependencias
COPY requirements.txt .

# Instala las dependencias listadas en requirements.txt
RUN pip install -r requirements.txt

# Descarga los recursos de NLTK (stopwords)
RUN python -m nltk.downloader stopwords

# Copia el resto de la aplicación al directorio de trabajo
COPY . .

# Expone el puerto 5000 (donde corre Flask)
EXPOSE 5000

# Comando a ejecutar cuando se inicie el contenedor
# Inicia la aplicación Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]