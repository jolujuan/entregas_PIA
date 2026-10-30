import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

#Leemos la imagen
image_path = 'data/test1.png'
img = cv2.imread(image_path)

#Instanciar al detectar el texto
reader = easyocr.Reader(['en'], gpu=False)

#Detectar el texto de la imagen
text_ = reader.readtext(img)
threshold=0.25

#Dibujar un cuadrado con el texto detectado
for t in text_:
    print(t)
    bbox,text,score=t
    if score>threshold:
        cv2.rectangle(img,bbox[0],bbox[2],(0,255,0),5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()