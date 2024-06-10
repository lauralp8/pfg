from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('video_frame')
def handle_video_frame(data):
    # El frame llega como una cadena de texto base64
    frame_data = data['frame']
    # Decodificar el frame base64 a un array numpy
    frame = np.frombuffer(base64.b64decode(frame_data), np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Aquí puedes procesar el frame usando tu modelo
    # Por ejemplo:
    # prediction = procesar_frame(frame, model)
def procesar_frame(frame, model):
    resized_frame = cv2.resize(frame, (224, 224))  # Ajusta el tamaño según lo que necesite tu modelo
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 224, 224, 3))  # Ajusta según el formato de entrada de tu modelo

    # Predicción del modelo
    prediction = model.predict(reshaped_frame)
    # Procesa la predicción según tus necesidades
    return prediction


    # Mostrar el frame en una ventana (opcional)
    cv2.imshow('Video', frame)
    cv2.waitKey(1)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
