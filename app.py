import re
from flask import Flask, render_template, request, jsonify, make_response, send_from_directory, url_for
from matplotlib import image
import numpy as np
import tensorflow as tf
import keras
import os
import cv2
from PIL import Image
import random
import string
import shutil

app = Flask(__name__)


def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def predict_from_video(mp4_path, output_folder, target_resolution=(426, 240)):
    model = keras.models.load_model('model.hdf5')

    cap = cv2.VideoCapture(mp4_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames_per_sequence = 5
    goal_sequences = []

    for start_frame in range(0, frame_count - frames_per_sequence + 1, frames_per_sequence):
        sequence_folder = os.path.join(
            output_folder, f"sequence_{int(start_frame/frames_per_sequence)}")
        create_directory(sequence_folder)

        images = []

        for offset in range(frames_per_sequence):
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, target_resolution)

            frame_filename = f"frame_{offset:02d}.jpg"
            frame_path = os.path.join(sequence_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            images.append(frame)

        file_paths = [
            os.path.join(sequence_folder, 'frame_00.jpg'),
            os.path.join(sequence_folder, 'frame_01.jpg'),
            os.path.join(sequence_folder, 'frame_02.jpg'),
            os.path.join(sequence_folder, 'frame_03.jpg'),
            os.path.join(sequence_folder, 'frame_04.jpg')]

        frames_list = [np.array(np.asarray(Image.open(file_path), dtype='bfloat16'))
                       for file_path in file_paths]

        frames_array = np.array(frames_list)
        frames_array = np.expand_dims(frames_array, axis=0)

        result = model.predict_on_batch(frames_array)

        if (result[0][1] > 0.6):
            print(start_frame/fps)
            goal_sequences.append([f"sequence_{int(start_frame/frames_per_sequence)}/frame_02.jpg", "%0.2f" % (
                start_frame/fps), "%0.2f" % ((start_frame+frames_per_sequence)/fps)])

    cap.release()
    return goal_sequences


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/output/<random_string>/<path:filename>')
def output_image(random_string, filename):
    return send_from_directory(os.path.join('output', random_string), filename)


@app.route('/upload', methods=['POST'])
def upload():
    random_string = generate_random_string(10)
    create_directory(f'uploads/{random_string}')
    create_directory(f'output/{random_string}')

    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    upload_file_path = os.path.join(
        "uploads", os.path.join(random_string, file.filename))
    output_file_path = os.path.join("output", random_string)

    file.save(upload_file_path)
    goal_sequences = predict_from_video(upload_file_path, output_file_path)
    if len(goal_sequences) == 0:
        return make_response("""
    <html>
        <head>
            <style>
                body {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                    font-family: 'Arial', sans-serif;
                }
                h1 {
                    text-align: center;
                }
                .back-button {
                    margin-top: 20px;
                    text-align: center;
                    display: block;
                }
                .back-button a {
                    display: inline-block;
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    transition: background-color 0.3s;
                }
                .back-button a:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <div>
            <h1>No goals found</h1> </div>
            <div class="back-button">
                <a href="/">Back to Main Page</a>
            </div>
        </body>
    </html>
""")

    html_table = """
<html>
<head>
    <style>
        table {
            width: 55%;
            border-collapse: collapse;
            margin: 20px;
            font-family: 'Arial', sans-serif;
        }
        th, td {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        img {
            max-width: 100%;
            max-height: 100%;
            display: block;
            margin: auto;
        }
        .back-button {
            margin-top: 20px;
            text-align: center;
        }
        .back-button a {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-family: 'Arial', sans-serif;
            transition: background-color 0.3s;
        }
        .back-button a:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
<div>
    <table>
        <tr>
            <th>Time (s)</th>
            <th>Frame</th>
        </tr>
"""

    for goal_sequence in goal_sequences:
        img_src = f"<img src='{url_for('output_image', random_string=random_string, filename=goal_sequence[0])}' alt='image'>"
        html_table += f"<tr><td>{goal_sequence[1]} - {goal_sequence[2]}</td><td>{img_src}</td></tr>"

    html_table += "</table></div>"

    html_table += """
    <div class="back-button">
        <a href="/">Back to Main Page</a>
    </div>
</body>
</html>
"""

    shutil.rmtree(f'uploads')
    # shutil.rmtree(f'output/{random_string}')

    return make_response(html_table)


if __name__ == '__main__':
    app.run(debug=True)
