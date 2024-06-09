from flask import Flask, Response
from gesture_recognition import recognize_gestures

app = Flask(__name__)


@app.route('/video_feed')
def video_feed():
    print("HOLAAAA")
    return Response(recognize_gestures(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)