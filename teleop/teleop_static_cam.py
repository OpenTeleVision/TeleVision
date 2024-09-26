"""
Stream the ZED's camera feed into the Vuer environment, which can then be viewed in a browser through the Quest.
"""

import time
from multiprocessing import Event, Queue, shared_memory

import cv2
import numpy as np
import pyzed.sl as sl
from pytransform3d import rotations

from constants_vuer import *
from tele_vision import OpenTeleVision

np.set_printoptions(precision=2, suppress=True)
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

# Capture 50 frames and stop
i = 0
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
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming)

try:
    while True:
        start = time.time()

        head_mat = (
            grd_yup2grd_zup[:3, :3] @ tv.head_matrix[:3, :3] @ grd_yup2grd_zup[:3, :3].T
        )
        if np.sum(head_mat) == 0:
            head_mat = np.eye(3)
        head_rot = rotations.quaternion_from_matrix(head_mat[0:3, 0:3])
        try:
            ypr = rotations.euler_from_quaternion(head_rot, 2, 1, 0, False)
        except:
            pass

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

        np.copyto(img_array, rgb)

        end = time.time()
except KeyboardInterrupt:
    # zed.close() causes some weirdness sometimes.
    pass
