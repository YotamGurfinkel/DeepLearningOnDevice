#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import pickle
import numpy.typing as npt
import argparse
from PIL import Image

PI_CAMERA_RES = {"width": 640, "height": 480, "dim": 3}
COLOR_PALETTE = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * COLOR_PALETTE
colors = (colors % 255).numpy().astype("uint8")


def get_args() -> tuple[str, str, int]:
    parser = argparse.ArgumentParser(
        description="Converts a pickled video file with the format [(raw_image_array, image_seg_array)] to a mp4 video",
        allow_abbrev=False,
    )

    parser.add_argument("--seg_size", type=int, default=224)
    parser.add_argument("pickled_file", type=str)
    parser.add_argument("out_video_file", type=str)
    args = parser.parse_args()

    assert args.out_video_file.endswith(".mp4"), "Expected out mp4 file"

    return args.pickled_file, args.out_video_file, args.seg_size


def get_colored_seg_image(image_seg_arr: npt.NDArray) -> Image.Image:
    img = Image.fromarray(image_seg_arr)
    img.putpalette(colors)
    return img


def main():
    pickled_file_path, out_video_file, seg_size = get_args()
    seg_res = (seg_size, seg_size)

    frames = []
    with open(pickled_file_path, "rb") as pickle_file:
        while True:
            try:
                frames += pickle.load(pickle_file)
            except (EOFError, pickle.UnpicklingError):
                break

    print([(i.shape, j.shape) for i, j in frames])
    assert all(
        raw_image.shape
        == (PI_CAMERA_RES["height"], PI_CAMERA_RES["width"], PI_CAMERA_RES["dim"])
        and seg_image.shape == seg_res
        for raw_image, seg_image in frames
    ), f"Expected raw image to be of dimension {PI_CAMERA_RES['width']}x{PI_CAMERA_RES['height']} and segmentation image to be of dimension {seg_res[0]}x{seg_res[1]}"

    video_writer = cv2.VideoWriter(
        out_video_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        2,
        (PI_CAMERA_RES["width"] * 2, PI_CAMERA_RES["height"]),
    )

    for raw_image_arr, seg_image in frames:
        raw_image = Image.fromarray(raw_image_arr.astype("uint8")).convert("RGBA")
        seg_image = (
            get_colored_seg_image(seg_image.astype("uint8"))
            .resize(
                (PI_CAMERA_RES["width"], PI_CAMERA_RES["height"]),
                resample=Image.Resampling.NEAREST,
            )
            .convert("RGBA")
        )

        combined_image = raw_image
        side_by_side_image = Image.new(
            "RGBA", (PI_CAMERA_RES["width"] * 2, PI_CAMERA_RES["height"])
        )
        side_by_side_image.paste(raw_image, (0, 0))

        seg_image.putalpha(50)

        combined_image.paste(seg_image, (0, 0), seg_image)
        side_by_side_image.paste(combined_image, (PI_CAMERA_RES["width"], 0))

        video_writer.write(np.array(side_by_side_image.convert("RGB")))

    video_writer.release()


if __name__ == "__main__":
    main()
