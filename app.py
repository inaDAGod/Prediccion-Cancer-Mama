from flask import Flask, render_template, request, redirect, url_for, flash
from model_j48 import train_and_predict  # Importar lógica del modelo J48
from model_cluster import process_clusters  # Importar lógica del Clustering
from model_neural import process_neural  # Importar lógica de la Red Neuronal
import os

app = Flask(__name__)
app.secret_key = "clave_secreta"
UPLOAD_FOLDER = './static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta de login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validar credenciales
        if username == "admin" and password == "admin":
            flash('Inicio de sesión exitoso', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Credenciales incorrectas. Inténtalo de nuevo.', 'danger')

    return render_template('login.html')

# Ruta para subir archivo
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            # Guardar el archivo subido
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            flash('Archivo subido correctamente', 'success')
            return redirect(url_for('model_options', file_path=file.filename))
        else:
            flash('Por favor, sube un archivo CSV válido.', 'danger')
    return render_template('upload.html')

@app.route('/contactos')
def contactos():
    return render_template('contactos.html')


# Ruta para mostrar opciones de modelo
@app.route('/model-options/<file_path>')
def model_options(file_path):
    return render_template('model_options.html', file_path=file_path)

# Ruta para ejecutar J48
@app.route('/run-model/J48/<file_path>', methods=['GET'])
def run_model_j48(file_path):
    try:
        # Ruta del archivo subido
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)

        # Procesar el archivo con J48
        prediction = train_and_predict(file_path)

        # Definir la interpretación
        if prediction == 2:
            interpretation = "Benigno"
        elif prediction == 4:
            interpretation = "Maligno"
        else:
            interpretation = "Desconocido"

        # Mostrar resultado, interpretación y gráfico
        return render_template(
            'result.html',
            result=interpretation,
            graph_path='uploads/tree.png',
            file_path=os.path.basename(file_path)
        )
    except Exception as e:
        flash(f"Error al procesar el modelo J48: {str(e)}", 'danger')
        return redirect(url_for('index'))

# Ruta para ejecutar Clustering
@app.route('/run-model/cluster/<file_path>', methods=['GET'])
def run_model_cluster(file_path):
    try:
        # Ruta del archivo subido
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)

        # Procesar el archivo con Clustering
        result = process_clusters(file_path)

        # Mostrar resultado y gráfico
        return render_template(
            'result_cluster.html',
            result=result,
            graph_path='uploads/clusters.png',
            file_path=os.path.basename(file_path)
        )
    except Exception as e:
        flash(f"Error al procesar el modelo de Clustering: {str(e)}", 'danger')
        return redirect(url_for('index'))

# Ruta para ejecutar Red Neuronal
@app.route('/run-model/neural/<file_path>', methods=['GET'])
def run_model_neural(file_path):
    try:
        # Ruta del archivo subido
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)

        # Procesar el archivo con Red Neuronal
        result = process_neural(file_path)

        # Mostrar resultado y gráficos
        return render_template(
            'result_neural.html',
            result=result,
            graph_path='uploads/neural_training.png',
            layers_graph_path='uploads/neural_layers.png',
            file_path=os.path.basename(file_path)
        )
    except Exception as e:
        # Extraer solo el nombre del archivo para redirigir
        file_name = os.path.basename(file_path)
        flash(f"Error al procesar el modelo de Red Neuronal: {str(e)}", 'danger')
        return redirect(url_for('model_options', file_path=file_name))

if __name__ == "__main__":
    app.run(debug=True)
