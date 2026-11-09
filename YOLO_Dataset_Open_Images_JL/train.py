from ultralytics import YOLO

# Cargar el modelo YOLO
model = YOLO("yolo26n.pt")

# Entrenar el modelo para detectar coches
results = model.train(
    data="configCar.yaml",
    epochs=5,
    batch=8
)
