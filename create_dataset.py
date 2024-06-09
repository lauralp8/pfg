import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

'''
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
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

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

'''
'''
import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio de datos
DATA_DIR = './data'

# Listas para almacenar los datos y etiquetas
data = []
labels = []

# Contador de imágenes procesadas
image_count = 0

def process_image(dir_, img_path):
    data_aux = []
    x_ = []
    y_ = []

    img_full_path = os.path.join(DATA_DIR, dir_, img_path)
    img = cv2.imread(img_full_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
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

        return data_aux, dir_
    return None, None

# Procesar imágenes y extraer características en lotes
batch_size = 1000  # Ajusta el tamaño del lote según tu memoria disponible

all_images = [(dir_, img) for dir_ in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, dir_))
              for img in os.listdir(os.path.join(DATA_DIR, dir_))]

for i in range(0, len(all_images), batch_size):
    batch = all_images[i:i + batch_size]
    for dir_, img_path in batch:
        try:
            data_aux, label = process_image(dir_, img_path)
            if data_aux is not None:
                data.append(data_aux)
                labels.append(label)
                image_count += 1

            if image_count % 100 == 0:
                print(f'{image_count} imágenes procesadas...')

        except Exception as e:
            print(f'Error procesando la imagen {os.path.join(DATA_DIR, dir_, img_path)}: {e}')

# Ajustar todas las secuencias de datos a la misma longitud
max_length = max(len(d) for d in data)
data_padded = np.array([d + [0] * (max_length - len(d)) for d in data])

# Guardar los datos en un archivo pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data_padded, 'labels': labels}, f)

print(f'Proceso completado. {image_count} imágenes fueron procesadas y guardadas en data.pickle')
'''
import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio de datos
DATA_DIR = './data'

# Listas para almacenar los datos y etiquetas
data = []
labels = []
image_sizes_original = []
image_sizes_preprocessed = []

# Contador de imágenes procesadas
image_count = 0

# Función de preprocesamiento de imágenes
def preprocess_image(img):
    # Redimensionar la imagen a un tamaño específico (por ejemplo, 256x256)
    img_resized = cv2.resize(img, (256, 256))
    # Normalizar la intensidad de píxeles (por ejemplo, escalar los valores de píxeles entre 0 y 1)
    img_normalized = img_resized.astype('float32') / 255.0
    return img_normalized
def process_image(dir_, img_path):
    data_aux = []
    x_ = []
    y_ = []

    img_full_path = os.path.join(DATA_DIR, dir_, img_path)
    img = cv2.imread(img_full_path)
    # Preprocesar la imagen
    img_preprocessed = preprocess_image(img)
    img_preprocessed = (img_preprocessed * 255).astype(np.uint8)  # Convertir de nuevo a uint8
    img_rgb = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
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

        return data_aux, dir_
    return None, None

# Procesar imágenes y extraer características en lotes
batch_size = 1000  # Ajusta el tamaño del lote según tu memoria disponible

all_images = [(dir_, img) for dir_ in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, dir_))
              for img in os.listdir(os.path.join(DATA_DIR, dir_))]

for i in range(0, len(all_images), batch_size):
    batch = all_images[i:i + batch_size]
    for dir_, img_path in batch:
        try:
            data_aux, label = process_image(dir_, img_path)
            if data_aux is not None:
                data.append(data_aux)
                labels.append(label)
                image_count += 1

            if image_count % 100 == 0:
                print(f'{image_count} imágenes procesadas...')

        except Exception as e:
            print(f'Error procesando la imagen {os.path.join(DATA_DIR, dir_, img_path)}: {e}')

# Calcular el tamaño promedio de las imágenes
avg_size_original = np.mean(image_sizes_original, axis=0)
avg_size_preprocessed = np.mean(image_sizes_preprocessed, axis=0)

# Guardar los datos en un archivo pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f'Proceso completado. {image_count} imágenes fueron procesadas y guardadas en data.pickle')


