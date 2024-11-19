import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def process_clusters(csv_path):
    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)

    # Preparar los datos (omitimos columnas no relevantes para el clustering)
    X = df.drop(columns=['Sample code number', 'Class'])

    # Aplicar K-Means con 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Reducir dimensiones para visualización con PCA (2D)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X)

    # Crear el gráfico con clusters
    save_cluster_graph(reduced_data, clusters)

    return "Clustering realizado con éxito."

def save_cluster_graph(reduced_data, clusters):
    plt.figure(figsize=(12, 8))

    # Colores y etiquetas para los clusters
    colors = ['#1f77b4', '#ff7f0e']
    labels = ['Posible Benigno', 'Posible Maligno']

    # Graficar cada cluster con colores y etiquetas
    for cluster in range(2):
        cluster_points = reduced_data[clusters == cluster]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            c=colors[cluster], label=labels[cluster],
            s=50, alpha=0.7
        )

    # Configuraciones del gráfico
    plt.title('Clustering con K-Means (PCA)', fontsize=16, fontweight='bold')
    plt.xlabel('Combinación Lineal 1 (PCA)', fontsize=12)
    plt.ylabel('Combinación Lineal 2 (PCA)', fontsize=12)
    plt.legend(title="Clusters", fontsize=10, loc='upper right')
    plt.grid(alpha=0.3)

    # Guardar el gráfico
    plt.savefig('./static/uploads/clusters.png')
    plt.close()
