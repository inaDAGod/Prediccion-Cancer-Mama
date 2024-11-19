import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def train_and_predict(csv_path):
    # Cargar los datos del archivo CSV
    df = pd.read_csv(csv_path)

    # Preparar los datos
    X = df.drop(columns=['Sample code number', 'Class'])
    y = df['Class']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo J48
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Generar un gráfico del árbol de decisión
    save_tree_graph(model, X.columns)

    # No hay predicción específica; procesamos el archivo
    return "Archivo procesado exitosamente."

# Guardar un gráfico del árbol de decisión
def save_tree_graph(model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=["Benigno", "Maligno"], filled=True)
    plt.savefig('./static/uploads/tree.png')
    plt.close()
