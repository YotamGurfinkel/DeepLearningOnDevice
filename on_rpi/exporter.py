#!/usr/bin/env python3

import tvm
import pickle
import time
import multiprocessing
import numpy.typing as npt
from skimage.measure import find_contours
from tvm.contrib.graph_executor import GraphModule
import numpy as np
import cv2
import RPi.GPIO as GPIO
import torchvision

MODEL_INPUT_NAME = "input0"
MODEL_FILE_NAME = "lib284.tar"
IMAGE_SIZE = 284
GPIO_LED_PORT = 4
PERSON_THRESHOLD = 3000
USE_LED = False

TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def prepare_image(image_array) -> tvm.nd.NDArray:
    transformed_tensor = TRANSFORM(image_array)
    return tvm.nd.array(transformed_tensor.numpy())


def prepare_model(lib_path) -> GraphModule:
    lib = tvm.runtime.load_module(lib_path)
    dev = tvm.cpu(0)
    graph = GraphModule(lib["default"](dev))
    return graph


def find_biggest_person(raw_prediction: npt.NDArray) -> float:
    """
    Given an array of class predictions, return the biggest person
    """
    if 15 not in raw_prediction:
        return 0

    person_predictions = raw_prediction.copy()
    person_predictions[person_predictions != 15] = 0
    contours = find_contours(person_predictions)
    return max(
        [
            cv2.contourArea(cv2.UMat(np.expand_dims(contour.astype(np.float32), 1)))
            for contour in contours
        ]
    )


def prepare_led():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(GPIO_LED_PORT, GPIO.OUT)


def run_model_export_video_data(
    event: multiprocessing.Event,
):
    if USE_LED:
        prepare_led()
    model = prepare_model(MODEL_FILE_NAME)
    video_stream = cv2.VideoCapture(0)

    video_data = []
    video_dump_file = open("video_data.dump", "wb")
    fps_values = []

    frames_passed = 0
    while not event.is_set():
        ret, frame = video_stream.read()
        if ret:
            start = time.time()
            image_tvm_array = prepare_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            model.set_input(MODEL_INPUT_NAME, image_tvm_array)
            model.run()
            raw_prediction = model.get_output(0).numpy()[0]
            class_predictions = raw_prediction.argmax(0)

            if USE_LED:
                GPIO.output(
                    GPIO_LED_PORT,
                    GPIO.HIGH
                    if 15 in class_predictions
                    and np.bincount(class_predictions.flatten())[15] > PERSON_THRESHOLD
                    else GPIO.LOW,
                )

            video_data.append(
                (frame.astype("uint8"), class_predictions.astype("uint8"))
            )
            if frames_passed == 5:
                pickle.dump(video_data, video_dump_file)
                video_data.clear()
                frames_passed = 0
            else:
                frames_passed += 1

            fps_values.append(1 / (time.time() - start))

    if video_data:
        pickle.dump(video_data, video_dump_file)

    if USE_LED:
        GPIO.output(GPIO_LED_PORT, GPIO.LOW)

    video_stream.release()
    video_dump_file.close()
    with open("avg_fps", "w") as f:
        f.write(str(np.array(fps_values).mean()))
