'''
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Cargar los datos
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Entrenar el modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Realizar predicciones
y_predict = model.predict(x_test)

# Calcular la precisión
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Guardar el modelo
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Generar y mostrar la matriz de confusión
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
disp.plot()
plt.title('Matriz de Confusión')
plt.show()

# Mostrar un informe de clasificación
print(classification_report(y_test, y_predict))

# Graficar la distribución de las clases en el conjunto de datos completo
plt.figure(figsize=(10, 6))
plt.hist(labels, bins=len(set(labels)), edgecolor='black')
plt.title('Distribución de las clases')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()

# Graficar la distribución de las clases en el conjunto de entrenamiento
plt.figure(figsize=(10, 6))
plt.hist(y_train, bins=len(set(y_train)), edgecolor='black')
plt.title('Distribución de las clases (Entrenamiento)')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()

# Graficar la distribución de las clases en el conjunto de prueba
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=len(set(y_test)), edgecolor='black')
plt.title('Distribución de las clases (Prueba)')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()
'''

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Cargar los datos
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializar y entrenar el modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predecir y calcular la precisión
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% de muestras fueron clasificadas correctamente!'.format(score * 100))

# Guardar el modelo
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

    # Generar y mostrar la matriz de confusión
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
disp.plot()
plt.title('Matriz de Confusión')
plt.show()

# Mostrar un informe de clasificación
print(classification_report(y_test, y_predict))

# Graficar la distribución de las clases en el conjunto de datos completo
plt.figure(figsize=(10, 6))
plt.hist(labels, bins=len(set(labels)), edgecolor='black')
plt.title('Distribución de las clases')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()

# Graficar la distribución de las clases en el conjunto de entrenamiento
plt.figure(figsize=(10, 6))
plt.hist(y_train, bins=len(set(y_train)), edgecolor='black')
plt.title('Distribución de las clases (Entrenamiento)')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()

# Graficar la distribución de las clases en el conjunto de prueba
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=len(set(y_test)), edgecolor='black')
plt.title('Distribución de las clases (Prueba)')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()
