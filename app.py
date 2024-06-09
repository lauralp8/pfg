from flask import Flask, Response
from gesture_recognition import recognize_gestures

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(recognize_gestures(camera_index=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
