#Importa los librerías necesarias
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("D:\Desktop\Proyecto Investigación\slim_shady"))
#Inicializa la lista de nombres conocidos y codificación conocida
knownEncodings = []
knownNames = []
#Navega sobre dirección de las imágenes
for (i, imagePath) in enumerate(imagePaths):
#Extrae el nombre de persona desde la dirección de las imágenes
 print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
 name = imagePath.split(os.path.sep)[-2]
#Carga la imágen de entrada y lo convierte a formato RGB para el tratamiento con Dlib (RGB)
 image = cv2.imread(imagePath)
 rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Detectar las coordenadas (x, y) de los rostros. Correspondiente a cada imagen.
 boxes = face_recognition.face_locations(rgb, model="cnn")
#Codifica cada rostro que se encuentra a la entrada
 encodings = face_recognition.face_encodings(rgb, boxes)
#Navega por los nombres codificados
 for encoding in encodings:
#Agrega a cada rostro un nombre y los codifica
  knownEncodings.append(encoding)
  knownNames.append(name)
#Guarda los rostros y nombres codificados en la dirección que se designe
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("D:\Desktop\Proyecto Investigación\mi_archivo.bin", "wb")
f.write(pickle.dumps(data))
f.close()