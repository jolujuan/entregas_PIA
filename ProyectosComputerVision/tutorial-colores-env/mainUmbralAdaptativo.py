import cv2
import numpy as np
from PIL import Image
from util import get_limits
yellow = [0, 255, 255]  # yellow in RGB colorspace


# Función para segmentar una imagen por color
def segment_by_color(image, reference_color):
    # Definir un rango de color cercano al color de referencia
    color_lower = np.array([reference_color[0] - 10, 50, 50])
    color_upper = np.array([reference_color[0] + 10, 255, 255])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, color_lower, color_upper)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image

#Color de referencia, amarillo en este caso
reference_color = np.array([30, 100, 100]) #Color HSV

# Crear ventanas separadas para mostrar las dos imágenes
cv2.namedWindow('Original', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Detected Color', cv2.WINDOW_AUTOSIZE)

# Mover la ventana 'Detected Color' a la derecha de la ventana 'Original'
cv2.moveWindow('Original', 0, 0)
cv2.moveWindow('Detected Color', 700, 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Segmentar la imagen por color
    segmented_image = segment_by_color(frame, reference_color)
    #Mostrar la imagen original y la imagen segmentada
    cv2.imshow('Original', frame)
    cv2.imshow('Detected Color', segmented_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberamos memoria después de salir del bucle
cap.release()
# eliminar todas las ventanas creadas
cv2.destroyAllWindows()

