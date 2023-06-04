from ctypes import *
import cv2
import darknet
from threading import Event, Lock, Thread
from queue import Queue
import numpy as np
import cv2
import urllib.request

WEIGHTS = "cfg/yolov4-tiny.weights"
CONFIG_FILE = "cfg/yolov4-tiny.cfg"
DATA_FILE = "data/coco.data"
THRESH = .25

FPS = 24
CAMERA_IP = "192.168.0.104"
CAMERA_URL = f"http://{CAMERA_IP}/cam-mid.jpg"
CAMERA_RUNNING = True

def inference(darknet_image_queue: Queue, detections_queue: Queue, finish_program: Event):
    while not finish_program.is_set():
        darknet_image = darknet_image_queue.get()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=THRESH)

        # For debugging
        with open("/debug/detections.txt", "a") as f:
            f.write(f"{detections}\n")
        
        darknet.free_image(darknet_image)

def image_getter(darknet_image_queue: Queue, finish_program: Event):
    while not finish_program.is_set():
        response = urllib.request.urlopen(CAMERA_URL)
        img_as_text = response.read()
        encoded_img = np.array(
            bytearray(img_as_text), dtype=np.uint8
        )
        img = cv2.imdecode(encoded_img, -1)

        # For debugging
        cv2.imwrite("/debug/data.jpg", img)

        # Sending image to darknet
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                    interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
        response.close()

if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)

    network, class_names, class_colors = darknet.load_network(
            CONFIG_FILE,
            DATA_FILE,
            WEIGHTS,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    print_lock = Lock()
    finish_program = Event()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, finish_program)).start()
    Thread(target=image_getter, args=(darknet_image_queue, finish_program)).start()