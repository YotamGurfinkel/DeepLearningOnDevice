#!/usr/bin/env python3

from torch import dist
import tvm
import numpy.typing as npt
from skimage.measure import find_contours
from tvm.contrib.graph_executor import GraphModule
import numpy as np
import cv2
import torchvision

MODEL_INPUT_NAME = "input0"
MODEL_FILE_NAME = "lib.tar"

TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
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


def run_model_export_video_data(event, biggest_person_pixels, found_bottle):
    model = prepare_model(MODEL_FILE_NAME)
    video_stream = cv2.VideoCapture(0)

    video_data = []

    while not event.is_set():
        ret, frame = video_stream.read()
        if ret:
            image_tvm_array = prepare_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            model.set_input(MODEL_INPUT_NAME, image_tvm_array)
            model.run()
            raw_prediction = model.get_output(0).numpy()[0]
            # video_data.append(frame)
            # video_data.append(raw_prediction)
            class_predictions = raw_prediction.argmax(0)
            biggest_person_value = find_biggest_person(class_predictions)
            # bottle_value = np.bincount(class_predictions.flatten())[5]
            with biggest_person_pixels.get_lock():
                biggest_person_pixels.value = biggest_person_value
            with found_bottle.get_lock():
                found_bottle.value = 5 in class_predictions

    video_stream.release()

    if video_data:
        np.savez("video_data.npz", *video_data)
