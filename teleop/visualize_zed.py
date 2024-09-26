"""
Visualize the ZED camera feed through a cv2 window.
Displays both the left and right camera feeds.
"""

import time
from multiprocessing import Event, Queue, shared_memory

import cv2
import numpy as np
import pyzed.sl as sl
from constants_vuer import *

resolution = (720, 1280)
crop_size_w = 1
crop_size_h = 0
resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = (
    sl.RESOLUTION.HD720
)  # Use HD720 opr HD1200 video mode, depending on camera type.
init_params.camera_fps = 60  # Set fps at 60


# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera Open : " + repr(err) + ". Exit program.")
    exit()

image_left = sl.Mat()
image_right = sl.Mat()
runtime_parameters = sl.RuntimeParameters()

img_shape = (resolution_cropped[0], 2 * resolution_cropped[1], 3)
img_height, img_width = resolution_cropped[:2]
shm = shared_memory.SharedMemory(
    create=True, size=np.prod(img_shape) * np.uint8().itemsize
)
img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)
image_queue = Queue()
toggle_streaming = Event()

done = False
while not done:
    start = time.time()

    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)

    bgr = np.hstack(
        (
            image_left.numpy()[crop_size_h:, crop_size_w:-crop_size_w],
            image_right.numpy()[crop_size_h:, crop_size_w:-crop_size_w],
        )
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)

    cv2.imshow("zed_left", image_left.numpy()[:, :, :3])
    cv2.imshow("zed_right", image_right.numpy()[:, :, :3])

    key = cv2.waitKey(1) & 0xFF

    end = time.time()

    if key == ord("q") or key == 27:
        done = True
zed.close()
