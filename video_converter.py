#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import numpy.typing as npt
import argparse
from PIL import Image
import voc

PI_CAMERA_RES = {"width": 640, "height": 480, "dim": 3}
SEG_RES = (21, 224, 224)
COLOR_PALETTE = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * COLOR_PALETTE
colors = (colors % 255).numpy().astype("uint8")


def get_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser(
        description="Converts a raw npz file with the format [(raw_image_array, image_seg_array)] to a mp4 video",
        allow_abbrev=False,
    )

    parser.add_argument("npz_file", type=str)
    parser.add_argument("out_video_file", type=str)
    args = parser.parse_args()

    assert args.npz_file.endswith(".npz"), "Expected npz file"
    assert args.out_video_file.endswith(".mp4"), "Expected out mp4 file"

    return args.npz_file, args.out_video_file


def get_colored_seg_image(image_seg_arr: npt.NDArray) -> Image.Image:
    pred = torch.tensor(image_seg_arr).argmax(0)
    img = Image.fromarray(pred.byte().cpu().numpy())
    img.putpalette(colors)
    return img


def main():
    npz_file_path, out_video_file = get_args()
    npz_file = np.load(npz_file_path)
    raw_video_data = npz_file.files

    frame_names = list(zip(raw_video_data[::2], raw_video_data[1::2]))

    assert all(
        npz_file[raw_image_name].shape
        == (PI_CAMERA_RES["height"], PI_CAMERA_RES["width"], PI_CAMERA_RES["dim"])
        and npz_file[seg_image_name].shape == SEG_RES
        for raw_image_name, seg_image_name in frame_names
    ), f"Expected raw image to be of dimension {PI_CAMERA_RES['width']}x{PI_CAMERA_RES['height']} and segmentation image to be of dimension {SEG_RES['width']}x{SEG_RES['height']}"

    video_writer = cv2.VideoWriter(
        out_video_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        6,
        (PI_CAMERA_RES["width"] * 2, PI_CAMERA_RES["height"]),
    )

    for raw_image_name, seg_image_name in frame_names:
        raw_image = Image.fromarray(npz_file[raw_image_name].astype("uint8")).convert(
            "RGBA"
        )
        seg_image = (
            get_colored_seg_image(npz_file[seg_image_name])
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

    npz_file.close()


if __name__ == "__main__":
    main()
