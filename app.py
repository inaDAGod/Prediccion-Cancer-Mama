from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = "clave_secreta"  # Clave necesaria para usar flash

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
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "admin":
            flash('Inicio de sesión exitoso', 'success')  # Mensaje para éxito
            return redirect(url_for('upload'))
        else:
            flash('Credenciales incorrectas', 'error')  # Mensaje para error
    return render_template('login.html')

# Ruta para subir archivo
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            flash('Archivo subido correctamente', 'success')
            return redirect(url_for('model_options'))
        else:
            flash('Error al subir el archivo', 'error')
    return render_template('upload.html')

# Ruta para mostrar opciones de modelo
@app.route('/model-options')
def model_options():
    return render_template('model_options.html')

# Ruta para ejecutar modelos
@app.route('/run-model/<model_name>')
def run_model(model_name):
    result = f"Ejecutando el modelo {model_name} con el archivo subido."
    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
