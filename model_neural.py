import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def process_neural(csv_path):
    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)

    # Preparar los datos
    X = df.drop(columns=['Sample code number', 'Class'])  # Características
    y = df['Class'].replace({2: 0, 4: 1})  # Etiquetas (2 -> Benigno, 4 -> Maligno)

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Construir la red neuronal
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Salida binaria (0 o 1)
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=0)

    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Guardar los gráficos
    save_training_graph(history)
    save_layer_graph([X_train.shape[1], 16, 8, 1])  # Dimensiones de las capas

    return f"Modelo entrenado con una precisión del {accuracy * 100:.2f}%."

def save_training_graph(history):
    import matplotlib
    matplotlib.use('Agg')  # Usar backend 'Agg' para evitar problemas en entornos web
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Gráfico de precisión
    plt.plot(history.history['accuracy'], label='Precisión Entrenamiento', color='blue')
    plt.plot(history.history['val_accuracy'], label='Precisión Validación', color='orange')
    plt.title('Precisión del Modelo durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(alpha=0.3)

    # Guardar el gráfico
    plt.savefig('./static/uploads/neural_training.png')
    plt.close()

def save_layer_graph(layer_sizes):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 6))

    # Coordenadas para las capas
    x_offsets = np.linspace(0.1, 0.9, len(layer_sizes))
    max_neurons = max(layer_sizes)

    for i, neurons in enumerate(layer_sizes):
        y_positions = np.linspace(0.2, 0.8, neurons)
        x_position = x_offsets[i]
        plt.scatter(
            [x_position] * neurons,
            y_positions,
            s=500,
            label=f'Capa {i + 1} ({neurons} neuronas)',
            alpha=0.7
        )
        if i > 0:
            # Conectar capas
            prev_y_positions = np.linspace(0.2, 0.8, layer_sizes[i - 1])
            for prev_y in prev_y_positions:
                for curr_y in y_positions:
                    plt.plot(
                        [x_offsets[i - 1], x_position],
                        [prev_y, curr_y],
                        color='gray',
                        alpha=0.3,
                    )

    # Configuración del gráfico
    plt.title('Arquitectura de la Red Neuronal', fontsize=16, fontweight='bold')
    plt.xlabel('Capas', fontsize=12)
    plt.ylabel('Neuronas', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(alpha=0.3)

    # Guardar el gráfico
    plt.savefig('./static/uploads/neural_layers.png')
    plt.close()
