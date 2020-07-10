"""
Detects and generates cropped images of faces using dlib

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
Modified by: Ayrton San Joaquin
"""
import os
import argparse
from os.path import join, basename
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Finds and crops the faces in a set of images in a folder ')
parser.add_argument('--i',type=str, default='.', help='input_path')
parser.add_argument('--o',type=str, default='../cropped_images',help='output_folder')


args = parser.parse_args()

if not os.path.isdir(args.i):
    os.mkdir(args.i)
if not os.path.isdir(args.o):
    os.mkdir(args.o)

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def test_full_image_network(video_path, output_path,
                            start_frame=0, end_frame=None):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    try:
        assert start_frame < num_frames - 1
    except AssertionError:
        print(start_frame, num_frames)
        continue
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            try:
              cv2.imwrite(join(output_path,'{}_{}.png'.format(basename(video_path),frame_num)), cropped_face)
            except:
              pass


video_path = args.i
output_path = args.o
if video_path.endswith('.mp4') or video_path.endswith('.avi'):
    test_full_image_network(video_path, output_path)
else:
    videos = os.listdir(video_path)
    for video in videos:
        video_path = join(video_path, video)
        test_full_image_network(video_path, output_path)