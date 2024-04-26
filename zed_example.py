import numpy as np
np.set_printoptions(precision=2, suppress=True)

import time
import cv2
from TeleVision import OpenTeleVision
import pyzed.sl as sl

grd_yup2grd_zup = np.array([[0, 0, -1, 0],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])

resolution = (720, 1280)

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 opr HD1200 video mode, depending on camera type.
init_params.camera_fps = 60  # Set fps at 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera Open : "+repr(err)+". Exit program.")
    exit()

image_left = sl.Mat()
image_right = sl.Mat()
runtime_parameters = sl.RuntimeParameters()
tv = OpenTeleVision(resolution)

while True :
    start = time.time()
    
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)

    rgb_left = cv2.cvtColor(image_left.numpy(), cv2.COLOR_BGRA2RGB)
    rgb_right = cv2.cvtColor(image_right.numpy(), cv2.COLOR_BGRA2RGB)
    
    tv.modify_shared_image(np.vstack((rgb_left, rgb_right)))
    
    end = time.time()
zed.close()
