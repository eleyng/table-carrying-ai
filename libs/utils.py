
import cv2
import os
from os.path import join
import re


def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def make_video(dir, name):
    """Make video from images in a folder.

    Args:
        dir (str): Path to folder containing images.
        name (str): Name of video to save.
    """
    image_folder = dir
    video_name = join(dir, f"{name}.avi")

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    sorted_images = sorted(images, key=natural_sort_key)
    frame = cv2.imread(os.path.join(image_folder, sorted_images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in sorted_images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()