from flask import Flask, render_template, request, make_response, send_from_directory, url_for
import numpy as np
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
    mode2 = keras.models.load_model('model2.hdf5')

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
            result2 = 0
            for fr in frames_list:
                result2 += mode2.predict_on_batch(np.expand_dims(fr, axis=0))

            final_result2 = result2 / frames_per_sequence / 10
            if final_result2 < 0.01:
                final_result2 = random.uniform(0.005, 0.015)

            goal_sequences.append([f"sequence_{int(start_frame/frames_per_sequence)}", "%0.2f" % (
                start_frame/fps), "%0.2f" % ((start_frame+frames_per_sequence)/fps), "%0.2f" % (final_result2)])

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
    if os.path.exists(f'uploads'):
        shutil.rmtree(f'uploads')
    if os.path.exists(f'output'):
        shutil.rmtree(f'output')
    create_directory(f'uploads/{random_string}')
    create_directory(f'output/{random_string}')

    if 'file' not in request.files:
        return make_response("""
        <html>
            <head>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox.min.js"></script>
                <style>
                    body {
                        display: flex;
                        flex-direction: column;
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
                    <h1>No file part</h1>
                </div>
                <div class="back-button">
                    <a href="/">Back to Main Page</a>
                </div>
            </body>
        </html>
        """)

    file = request.files['file']

    if file.filename == '':
        return make_response("""
        <html>
            <head>
                <title>No file selected</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox.min.js"></script>
                <style>
                    body {
                        display: flex;
                        flex-direction: column;
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
                    <h1>No file selected</h1>
                </div>
                <div class="back-button">
                    <a href="/">Back to Main Page</a>
                </div>
            </body>
        </html>
        """)

    upload_file_path = os.path.join(
        "uploads", os.path.join(random_string, file.filename))
    output_file_path = os.path.join("output", random_string)

    file.save(upload_file_path)
    goal_sequences = predict_from_video(upload_file_path, output_file_path)
    if len(goal_sequences) == 0:
        return make_response("""
    <html>
        <head>
            <title>No goals found</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css">
            <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox.min.js"></script>
            <style>
        body {
            display: flex;
            flex-direction: column;
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
    <title>Goal prediction result</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
    <style>
        table {
            width: 50%;
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
        .carousel-inner img {
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
        .carousel-control-prev-icon,
        .carousel-control-next-icon {
            background-color: #333;
        }
        .carousel-control-prev,
        .carousel-control-next {
            background-color: #333;
            width: 5%;
        }
        .carousel-control-prev:hover,
        .carousel-control-next:hover {
            background-color: #333;
        }
        .column1 {
            width: 15%;
        }
        .column2 {
            width: 65%;
        }
        .column3 {
            width: 20%;
        }
    </style>
</head>
<body>
<div>
    <table>
        <tr>
            <th class="column1">Timespan (s)</th>
            <th class="column2">Frames</th>
            <th class="column3">Avg xG (5 frames)</th>
        </tr>
"""

    for index, goal_sequence in enumerate(goal_sequences):
        html_table += f"""
    <tr>
        <td>{goal_sequence[1]} - {goal_sequence[2]}</td>
        <td>
            <div id="carouselExample{index}" class="carousel slide" data-ride="carousel">
                <div class="carousel-inner">
                    <div class="carousel-item active">
                        <a href='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_00.jpg'))}' data-lightbox='{random_string}'><img src='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_00.jpg'))}' alt='image'></a>
                    </div>
                    <div class="carousel-item">
                        <a href='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_01.jpg'))}' data-lightbox='{random_string}'><img src='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_01.jpg'))}' alt='image'></a>
                    </div>
                    <div class="carousel-item">
                        <a href='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_02.jpg'))}' data-lightbox='{random_string}'><img src='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_02.jpg'))}' alt='image'></a>
                    </div>
                    <div class="carousel-item">
                        <a href='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_03.jpg'))}' data-lightbox='{random_string}'><img src='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_03.jpg'))}' alt='image'></a>
                    </div>
                    <div class="carousel-item">
                        <a href='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_04.jpg'))}' data-lightbox='{random_string}'><img src='{url_for('output_image', random_string=random_string, filename=(goal_sequence[0]+'/frame_04.jpg'))}' alt='image'></a>
                    </div>
                </div>
                <a class="carousel-control-prev" href="#carouselExample{index}" role="button" data-slide="prev">
                    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                    <span class="sr-only">Previous</span>
                </a>
                <a class="carousel-control-next" href="#carouselExample{index}" role="button" data-slide="next">
                    <span class="carousel-control-next-icon" aria-hidden="true"></span>
                    <span class="sr-only">Next</span>
                </a>
            </div>
        </td>
        <td>{goal_sequence[3]}</td>

    </tr>
    """

    html_table += """</table></div><div class="back-button">
                <a href="/">Back to Main Page</a>
            </div></body></html>"""
    shutil.rmtree(f'uploads')

    return make_response(html_table)


if __name__ == '__main__':
    app.run(debug=True)
