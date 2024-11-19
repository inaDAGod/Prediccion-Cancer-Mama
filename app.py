from flask import Flask, render_template, request, redirect, url_for, flash
from model_j48 import train_and_predict  # Importar lógica del modelo J48
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

        # Mostrar resultado y gráfico
        return render_template('result.html', result=f"El modelo J48 procesó el archivo con éxito.", graph_path='uploads/tree.png')
    except Exception as e:
        flash(f"Error al procesar el modelo J48: {str(e)}", 'danger')
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
