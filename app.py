from flask import Flask, render_template, request
from flask import send_from_directory
import numpy as np
import tensorflow as tf
import keras

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Save the file to a folder of your choice
    file.save(f'uploads/{file.filename}')
    model = keras.models.load_model('model.hdf5')
    print(model.weights)

    
    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)