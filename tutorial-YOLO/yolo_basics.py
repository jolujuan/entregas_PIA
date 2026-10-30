from ultralytics import YOLO
import numpy

# Cargar el modelo preentrenado de YOLO
# Si no lo tenemos en local lo descargara
model = YOLO("yolo11n.pt")

# predict sobre una imagen "inference/images/img0.JPG" indicando que detecte solo
# si estamos un 25% seguros como mínimo y guardamos resultado en disco (save=True)
detection_output = model.predict(source="inference/images/img0.JPG", conf=0.25, save=True)

# Imprimir información del tensor
print(detection_output)

# Si se esta utilizando GPU (cuda) se debe pasar la
# información antes a la CPU para poder convertir a numpy
if model.device.type == 'cuda':
    print(detection_output[0].cpu().numpy())
else:
    print(detection_output[0].numpy())