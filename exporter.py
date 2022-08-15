#!/usr/bin/env python3

import tvm
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


def run_model_export_video_data(event):
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
            video_data.append(frame)
            video_data.append(raw_prediction)

    video_stream.release()

    if video_data:
        np.savez("video_data.npz", *video_data)
