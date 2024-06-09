from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

def send_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede leer el frame de la cámara")
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        socketio.emit('frame', {'image': frame})

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.start_background_task(send_frame)
    socketio.run(app, host='0.0.0.0', port=5000)
