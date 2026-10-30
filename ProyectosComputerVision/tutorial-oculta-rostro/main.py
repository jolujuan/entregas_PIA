import cv2
import os
import argparse

def process_img(img, detector):
    #Indicar el tamaño real de la imagen a procesar
    H, W, _ = img.shape
    detector.setInputSize((W, H))

    #Ejecutar la detección
    _, faces = detector.detect(img)

    if faces is not None:
        for face in faces:
            x1, y1, w, h = face[0:4]
            score = face[14]
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            w = min(w, W - x1)
            h = min(h, H - y1)

            #Difuminar el rostro
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))

            #Dibujar el rectangulo verde
            #img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)

            #Escribir el score encima del rectangulo
            #texto = f"Score: {score:.2f}"
            #cv2.putText(img, texto, (x1, y1 - 10),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

args = argparse.ArgumentParser()
args.add_argument('--mode', default='video')
args.add_argument('--filePath', default='./data/testVideo.mp4')
args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
#Detectar caras
detector = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000
)
if args.mode in ["image"]:
    #Leer la imagen
    img = cv2.imread(args.filePath)
    img = process_img(img, detector)

    # Mostrar imagen en una ventana
    cv2.imshow('Rostros difuminados', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
elif args.mode in ['video']:
    cap = cv2.VideoCapture(args.filePath)

    # btener los frames por segundo
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #Leemos primer frame
    ret, frame = cap.read()

    # rear Video de salida (output_dir/output.mp4) donde guardaremos el video
    #con codificación MP4v a fps frames por segundo y el tamaño de video lo
    #sacamos del tamaño del video original
    output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                   cv2.VideoWriter_fourcc(*'MP4V'),
                                   fps,
                                   (frame.shape[1], frame.shape[0]))

    # Iteramos en todos los frames del video original
    while ret:
        #Modificamos el frame para ocultar rostros
        newFrame = process_img(frame, detector)

        #Añadimos el frame el modificado, no el original)
        output_video.write(newFrame)

        #Vamos mostrando lo que vamos procesando
        cv2.imshow('frame', newFrame)
        cv2.waitKey(1)

        #Leemos el siguiente frame
        ret, frame = cap.read()

    # Liberamos memoria
    cap.release()
    output_video.release()

elif args.mode in ['webcam']:
    cap = cv2.VideoCapture(0)
    fps = 25
    ret, frame = cap.read()

    while ret:
        #Modificamos el frame para ocultar rostros
        newFrame = process_img(frame, detector)

        #Vamos mostrando lo que vamos procesando
        cv2.imshow('frame', newFrame)

        #Salimos del bucle si se pulsa la tecla ESC cdigo 27
        if cv2.waitKey(fps) & 0xFF == 27:
            break

        #Leemos el siguiente frame
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

       

