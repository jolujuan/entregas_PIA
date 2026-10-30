import numpy as np
from ultralytics import YOLO
import cv2
import random
import yt_dlp
import subprocess

# Abrir fichero donde se especifican los
# nombres de los distintos objetos
fileCocoObjects = open("utils/coco.txt", "r")
# Leer fichero
data = fileCocoObjects.read()
# Obtenemos objetos separado por newline (\n)
class_list = data.split("\n")
fileCocoObjects.close()
#print(class_list)

# Generar colores aleatorios para cada objeto
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))
    
# Cargar modelo preentrenado YOLO
model = YOLO("weights/yolo11n.pt")

# Vals to resize video frames | small frame optimise the run
frame_wid = 1280
frame_hyt = 720



#Leer la URL del video mediante la libreria yt_dlp
url = "https://www.youtube.com/watch?v=6BPuGeS6O4w"

#Utilizamos el formato 135 oara detectar objetos con YOLO sin 
#procesar tanta resolución
opciones = {
    "format": "298",
    "quiet": True
}

with yt_dlp.YoutubeDL(opciones) as ydl:
    info = ydl.extract_info(url, download=False)
    video_url = info["url"]
    # Obtenemos la URL del audio original del video
    audio_url = next(
        formato["url"]
        for formato in info["formats"]
        if formato["format_id"] == "140"
    )   
 
cap = cv2.VideoCapture(video_url)
 
       
#cap = cv2.VideoCapture("inference/videos/afriq0.MP4") #Video local
#cap = cv2.VideoCapture(0) #Webcam

if not cap.isOpened():
    print("No se ha podido abrir el video/camara")
    exit()

# Obtenemos los FPS del video original
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 60

# Creamos el archivo para guardar el video procesado
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    "video_procesado.mp4",
    fourcc,
    fps,
    (frame_wid, frame_hyt)
)

if not writer.isOpened():
    raise RuntimeError("No se ha podido crear el archivo de video")

while True:
    # Capturar frame a frame
    ret, frame = cap.read()
    # ret es verdadero si se ha leido correctamente el frame

    if not ret:
        print("No se ha podido leer el 'frame'. ¿Hemos llegado al final? Saliendo ...")
        break

    # Procesar frame
    # Optimizacion opcional: Redimensionamos a un tamaño pequeño
    frame = cv2.resize(frame, (frame_wid, frame_hyt))
    # Predict sobre la imagen
    detect_params = model.predict(source=[frame], conf=0.35, imgsz=1280, save=False)
    # Si se esta utilizando GPU cuda se debe pasar la
    # información antes a la CPU para poder convertir a numpy
    parametrosDetectados = []
    if model.device.type == 'cuda':
        parametrosDetectados = detect_params[0].cpu()
    else:
        parametrosDetectados = detect_params[0].cpu()
        
    #print(parametrosDetectados.numpy())  # Impresión parametros detectados
    if len(parametrosDetectados.numpy()) != 0:  # Si es 0 no se ha detectado nada
        for objetoDetectado in parametrosDetectados.boxes:  # Iteramos boxes
            # clsID = int(objetoDetectado.cls.nunmpy()[0])  # CUANDO NO TENEMOS GPU UTILIZAMOS ESTA LINEA

            clsID = int(objetoDetectado.cls.numpy()[0])  # clase del objeto (numero)
            conf = round(objetoDetectado.conf.numpy()[0], 3)  # probabilidad/confianza
            x1, y1, x2, y2 = objetoDetectado.xyxy.numpy()[0]  # bounding box
            cv2.rectangle(
                frame,  # imagen sobre la que dibujar rectangulo
                (int(x1), int(y1)),  # x,y del rectangulo deben ser enteros
                (int(x2), int(y2)),  # x2,y2 ancho, y2=alto
                detection_colors[clsID],  # color de la clase de objeto
                3,
            )

            # Dibujar el nombre de la clase (no el número) y la probabilidad
            font = cv2.FONT_HERSHEY_COMPLEX
            print (class_list[int(clsID)])
            cv2.putText(
                frame,
                #class_list[int(clsID)] + " " + str(conf) + "%",
                class_list[int(clsID)],

                (int(x1), int(y1) - 10),  # coordenadas númericas no decimales
                font,
                1,
                (255, 255, 255),
                2,
            )

    
    # Guardamos el frame con los objetos detectados
    writer.write(frame)
    
    # Mostrar la imagen resultado
    cv2.imshow("ObjectDetection", frame)
    
    # Finalizar si se pulsa una "q"
    if cv2.waitKey(20) == ord("q"):
        break
    # fin procesamiento

# Cuando hemos finalizado liberar la captura
cap.release()
writer.release()
cv2.destroyAllWindows()

# Unimos el video procesado con el audio original
subprocess.run([
    "ffmpeg",
    "-y",
    "-i", "video_procesado.mp4",
    "-i", audio_url,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "copy",
    "-c:a", "aac",
    "-shortest",
    "resultado_final.mp4"
], check=True)

print("Video guardado correctamente con audio")