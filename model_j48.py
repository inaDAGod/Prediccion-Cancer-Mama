import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def train_and_predict(csv_path):
    # Carga de datos
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Sample code number', 'Class'])
    y = df['Class']  # 2 = Benigno, 4 = Maligno

    # División en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Entrenamiento
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Guarda el árbol
    save_tree_graph(model, X.columns)

    # Predice el primer ejemplo de test
    pred = model.predict(X_test.iloc[[0]])[0]
    return pred  # Devuelve 2 o 4

def save_tree_graph(model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(model,
              feature_names=feature_names,
              class_names=["Benigno", "Maligno"],
              filled=True)
    plt.savefig('./static/uploads/tree.png')
    plt.close()

