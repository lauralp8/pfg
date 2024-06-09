from flask import request, jsonify, render_template
import cv2
import numpy as np
import base64
import pickle
import mediapipe as mp

app = Flask(__name__)

# Cargar el modelo
model_dict = pickle.load(open('./model.p', 'rb'))
modelo = model_dict['model']

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Diccionario de etiquetas
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Longitud de características esperadas (84 en este caso)
expected_feature_length = 84

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Asegurarse de que las características tengan la longitud esperada
        if len(data_aux) < expected_feature_length:
            data_aux.extend([0] * (expected_feature_length - len(data_aux)))  # Rellenar con ceros
        elif len(data_aux) > expected_feature_length:
            data_aux = data_aux[:expected_feature_length]  # Truncar

        prediction = modelo.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        return jsonify({'prediction': predicted_character})

    return jsonify({'prediction': 'No se detectaron manos'})

def recognize_gestures():
    # Aquí puedes poner la lógica para reconocer gestos y devolver los frames
    # Por ejemplo, podrías capturar video desde una cámara, procesarlo para detectar gestos,
    # y luego devolver los frames en formato de bytes para ser mostrados en la página web.
    # Aquí tienes un ejemplo básico utilizando OpenCV para capturar video desde la cámara:

    cap = cv2.VideoCapture(0)  # Abrir la cámara
    while True:
        ret, frame = cap.read()  # Leer un fotograma desde la cámara
        if not ret:
            break

        # Aquí podrías agregar tu lógica para detectar gestos en el fotograma
        # y dibujar cualquier información adicional en el fotograma antes de devolverlo.

        # Convierte el fotograma a bytes para ser enviado como respuesta HTTP
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Devolver el fotograma como bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()  # Liberar la cámara cuando hayamos terminado

