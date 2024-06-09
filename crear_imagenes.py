import os
import cv2

data = './data'

if not os.path.exists(data):
    os.makedirs(data)

number_classes = 26
dataset_size = 300

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: no se pudo abrir la cámara.")
else:
    for j in range(number_classes):
        if not os.path.exists(os.path.join(data, str(j))):
            os.makedirs(os.path.join(data, str(j)))

        print('Collecting data for class {}'.format(j))

        done = False
        while True: 
            ret, frame = cap.read()
            cv2.putText(frame, 'Presiona la letra "Q" para empezar', (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break
        
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(data, str(j), '{}.jpg'.format(counter)), frame)

            counter += 1
            if cv2.waitKey(1) == 27:  # Presiona 'ESC' para salir temprano si es necesario
                    break
    
    cap.release()
    cv2.destroyAllWindows()
