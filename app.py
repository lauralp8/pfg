import os
import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from gesture_recognition import recognize_gestures

app = Flask(__name__)

# Cargar el modelo de reconocimiento de gestos
model = load_model('model.h5')

# Dirección IP de la cámara remota
camera_ip = 'http://192.168.1.100:8080/video'

def gen_frames():
    # Intenta abrir la cámara remota
    cap = cv2.VideoCapture(camera_ip)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Realiza el reconocimiento de gestos en el frame
            frame = recognize_gestures(frame, model)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
