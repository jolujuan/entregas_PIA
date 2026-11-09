from ultralytics import YOLO
import numpy

import os
IMAGEN_DIR = os.path.join('.', 'imagenes')
imagen_path = os.path.join(IMAGEN_DIR, 'imagenAlpaca.jpg')
imagen_path_out = '{}_out.jpg'.format(imagen_path)
model_path = os.path.join('.', 'runs', 'detect', 'train-3', 'weights', 'last.pt')

# Cargar el modelo preentrenado YOLO
model = YOLO(model_path)  # load a custom model

# predict sobre una imagen "inference/images/img0.JPG", indicando que detecte solo
# si estamos un 80% seguros como mínimo y guardamos resultado en disco (save=True)
detection_output = model.predict(source=imagen_path, conf=0.75, save=True)
# Imprimir la información siguiente es opcional y no influye en la predicción:
# Imprimir información del tensor
print(detection_output)

# Si se esta utilizando GPU (cuda) se debe pasar la
# información antes a la CPU para poder convertir a numpy
if model.device.type == 'cuda':
    print(detection_output[0].cpu().numpy())
else:
    print(detection_output[0].numpy())