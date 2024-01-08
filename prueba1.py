import cv2
import dlib

# Carga el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

# Carga el modelo preentrenado para la detección de puntos faciales
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializa la cámara
cap = cv2.VideoCapture(0)

while True:
    # Captura el fotograma de la cámara
    ret, frame = cap.read()

    # Convierte el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostros en el fotograma
    faces = detector(gray)

    for face in faces:
        # Dibuja un rectángulo alrededor del rostro
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Muestra el fotograma con las detecciones
    cv2.imshow("Facial Recognition", frame)

    # Sale del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()