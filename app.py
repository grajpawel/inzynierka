from tabnanny import verbose
from flask import Flask, render_template, request
from flask import send_from_directory
import numpy as np
import tensorflow as tf
import keras
import os
import cv2
from PIL import Image

app = Flask(__name__)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def split_mp4_into_frame_sequences(mp4_path, output_folder, target_resolution=(426, 240)):
    model = keras.models.load_model('model.hdf5')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(mp4_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_per_sequence = 5

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

        # file_paths = [
        #     os.path.join(sequence_folder, 'frame_00.jpg'),
        #     os.path.join(sequence_folder, 'frame_01.jpg'),
        #     os.path.join(sequence_folder, 'frame_02.jpg'),
        #     os.path.join(sequence_folder, 'frame_03.jpg'),
        #     os.path.join(sequence_folder, 'frame_04.jpg')]

        # frames_list = [np.array(np.asarray(Image.open(file_path), dtype='bfloat16'))
        #                for file_path in file_paths]

        frames_array = np.array(images)
        frames_array = np.expand_dims(frames_array, axis=0)

        # keras_list = keras.utils.image_dataset_from_directory(
        #     sequence_folder,
        #     labels=None,
        #     label_mode="int",
        #     class_names=None,
        #     color_mode="rgb",
        #     batch_size=1,
        #     image_size=(240, 426),
        #     shuffle=False,
        #     seed=None,
        #     validation_split=None,
        #     subset=None,
        #     interpolation="bilinear",
        #     follow_links=False,
        #     crop_to_aspect_ratio=False)

        # keras_list = np.expand_dims(keras_list, axis=0)

        result = model.predict(frames_array, verbose=0)

        if (result[0][1] > 0.6):
            print(result)

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    create_directory('uploads')
    create_directory('output')

    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file.save(f'uploads/{file.filename}')
    split_mp4_into_frame_sequences(f'uploads/{file.filename}', 'output')

    return '<h1 style="color: #000; text-align: center; width: 100%; font-family: \'Arial\', sans-serif;">File uploaded successfully</h1>'


if __name__ == '__main__':
    app.run(debug=True)
