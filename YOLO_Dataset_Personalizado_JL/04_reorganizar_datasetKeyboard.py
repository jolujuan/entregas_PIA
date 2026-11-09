
import os
import shutil
import random

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGEN = os.path.join(BASE_DIR, 'data_keyboard')
DESTINO = os.path.join(BASE_DIR, 'data')

# Comprobar las carpetas
if not os.path.isdir(ORIGEN):
    raise FileNotFoundError(f'No existe: {ORIGEN}')

# Buscar imágenes con sus anotaciones correspondientes
imagenes = []

for archivo in sorted(os.listdir(ORIGEN)):
    nombre, extension = os.path.splitext(archivo)

    if extension.lower() in ('.jpg', '.jpeg', '.png'):
        etiqueta = nombre + '.txt'

        if os.path.isfile(os.path.join(ORIGEN, etiqueta)):
            imagenes.append((archivo, etiqueta))
        else:
            print(f'AVISO: Sin etiqueta: {archivo}')

if not imagenes:
    raise RuntimeError('No hay imágenes con etiquetas.')

# Evitar sobrescribir un dataset existente
if os.path.exists(DESTINO):
    raise RuntimeError(
        'La carpeta data ya existe. '
        'Revísala antes de volver a generar el dataset.'
    )

# Mezclar las imágenes
random.seed(42)
random.shuffle(imagenes)

# Distribución 80% entrenamiento, 15% validación, 5% prueba
total = len(imagenes)
n_train = int(total * 0.80)
n_val = int(total * 0.15)

grupos = {
    'train': imagenes[:n_train],
    'validation': imagenes[n_train:n_train + n_val],
    'test': imagenes[n_train + n_val:]
}

# Crear las carpetas y organizar los archivos
for grupo, archivos in grupos.items():

    carpeta_imagenes = os.path.join(DESTINO, 'images', grupo)
    carpeta_etiquetas = os.path.join(DESTINO, 'labels', grupo)

    os.makedirs(carpeta_imagenes, exist_ok=True)
    os.makedirs(carpeta_etiquetas, exist_ok=True)

    for imagen, etiqueta in archivos:
        # Generar un nombre único y sencillo
        numero = len(os.listdir(carpeta_imagenes)) + 1
        nombre_nuevo = f'keyboard_{grupo}_{numero:03d}'

        extension = os.path.splitext(imagen)[1].lower()

        shutil.copy2(
            os.path.join(ORIGEN, imagen),
            os.path.join(carpeta_imagenes, nombre_nuevo + extension)
        )

        shutil.copy2(
            os.path.join(ORIGEN, etiqueta),
            os.path.join(carpeta_etiquetas, nombre_nuevo + '.txt')
        )

    print(f'{grupo}: {len(archivos)} imágenes y etiquetas')

print(f'\nTotal: {total} imágenes')
print('Dataset organizado correctamente para YOLO.')
