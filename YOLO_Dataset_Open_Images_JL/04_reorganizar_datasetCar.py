
import os
import shutil
import random

# Ruta del dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data_car')

# Carpetas originales
IMG_DIR = os.path.join(DATA_DIR, 'train', 'imgs')
ANN_DIR = os.path.join(DATA_DIR, 'train', 'anns')

# Obtener imágenes con sus etiquetas correspondientes
imagenes = []

for archivo in os.listdir(IMG_DIR):
    if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        nombre = os.path.splitext(archivo)[0]
        etiqueta = nombre + '.txt'

        if os.path.isfile(os.path.join(ANN_DIR, etiqueta)):
            imagenes.append((archivo, etiqueta))

# Mezclar de manera reproducible
random.seed(42)
random.shuffle(imagenes)

total = len(imagenes)
n_train = int(total * 0.80)
n_val = int(total * 0.15)

grupos = {
    'train': imagenes[:n_train],
    'validation': imagenes[n_train:n_train + n_val],
    'test': imagenes[n_train + n_val:]
}

# Crear carpetas y copiar los archivos
for grupo, archivos in grupos.items():
    destino_imgs = os.path.join(DATA_DIR, 'images', grupo)
    destino_anns = os.path.join(DATA_DIR, 'labels', grupo)

    os.makedirs(destino_imgs, exist_ok=True)
    os.makedirs(destino_anns, exist_ok=True)

    for imagen, etiqueta in archivos:
        shutil.copy2(
            os.path.join(IMG_DIR, imagen),
            os.path.join(destino_imgs, imagen)
        )

        shutil.copy2(
            os.path.join(ANN_DIR, etiqueta),
            os.path.join(destino_anns, etiqueta)
        )

    print(f'{grupo}: {len(archivos)} imágenes y etiquetas')

print('Distribución completada')
print('Conserva las carpetas originales hasta verificar el resultado.')
