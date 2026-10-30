import cv2
import numpy as np
from PIL import Image
from util import get_limits
yellow = [0, 255, 255]  # yellow in RGB colorspace

cap = cv2.VideoCapture(0)
# Crear ventanas separadas para mostrar las dos imágenes
cv2.namedWindow('Original', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Detected Color', cv2.WINDOW_AUTOSIZE)

# Mover la ventana 'Detected Color' a la derecha de la ventana 'Original'
cv2.moveWindow('Original', 0, 0)
cv2.moveWindow('Detected Color', 700, 0)

while True:
    ret, frame = cap.read()
    
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=yellow)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    # Crear una imagen negra del mismo tamaño que la imagen original
    black_background = np.zeros_like(frame)
    # Copiar solo los píxeles detectados en la imagen original
    black_background[mask != 0] = frame[mask != 0]
    
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    #Mostrar la imagen original y la imagen segmentada
    cv2.imshow('Original', frame)
    cv2.imshow('Detected Color', black_background)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberamos memoria después de salir del bucle
cap.release()
# eliminar todas las ventanas creadas
cv2.destroyAllWindows()

