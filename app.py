from flask import Flask, render_template, request
from flask import send_from_directory
import numpy as np
import tensorflow as tf
import keras
import os
import cv2

app = Flask(__name__)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def split_mp4_into_frame_sequences(mp4_path, output_folder, target_resolution=(1280, 720)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the MP4 file
    cap = cv2.VideoCapture(mp4_path)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the number of frames per sequence
    frames_per_sequence = 3

    # Read frames and save sequences of 3 consecutive frames into folders
    for start_frame in range(0, frame_count - frames_per_sequence + 1, frames_per_sequence):
        sequence_folder = os.path.join(
            output_folder, f"sequence_{start_frame}")
        os.makedirs(sequence_folder)

        for offset in range(frames_per_sequence):
            ret, frame = cap.read()

            if not ret:
                break

            # Resize frame to the target resolution
            frame = cv2.resize(frame, target_resolution)

            frame_filename = f"frame_{offset:02d}.jpg"
            frame_path = os.path.join(sequence_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

    # Release the video capture object
    cap.release()


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
    create_directory('uploads')
    create_directory('output')
    file.save(f'uploads/{file.filename}')
    model = keras.models.load_model('model.hdf5')
    # print(model.weights)
    split_mp4_into_frame_sequences(f'uploads/{file.filename}', 'output')

    return 'File uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True)
