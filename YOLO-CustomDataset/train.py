from ultralytics import YOLO

# Carga un modelo YOLO 
model = YOLO("yolo26n.pt") # Versión del año correspondiente
results = model.train(data="config.yaml", 
                      epochs=5,
                      batch=32)

