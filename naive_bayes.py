import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Cargar los datos
data_dict = pickle.load(open('data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Entrenar el modelo Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

# Realizar predicciones
y_predict_nb = nb_model.predict(x_test)

# Calcular la precisión
score_nb = accuracy_score(y_predict_nb, y_test)
print(f'{score_nb * 100:.2f}% of samples were classified correctly with Naive Bayes!')

# Generar y mostrar la matriz de confusión
cm_nb = confusion_matrix(y_test, y_predict_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=np.unique(labels))
disp_nb.plot()
plt.title('Matriz de Confusión - Naive Bayes')
plt.show()

# Mostrar un informe de clasificación
print(classification_report(y_test, y_predict_nb))
