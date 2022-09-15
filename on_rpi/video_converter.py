#!/usr/bin/env python3

import cv2
import torch
from PIL import Image
import numpy as np
import pickle
import numpy.typing as npt
import argparse

PI_CAMERA_RES = {"width": 640, "height": 480, "dim": 3}
COLOR_PALETTE = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
WANTED_CLASSES = [5, 15]

colors = torch.as_tensor([i for i in range(21)])[:, None] * COLOR_PALETTE
colors = (colors % 255).numpy().astype("uint8")

CLASSES_NAMES = {
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}


def get_args() -> tuple[str, str, int, bool, int, bool]:
    parser = argparse.ArgumentParser(
        description="Converts a pickled video file with the format [(raw_image_array, image_seg_array)] to a mp4 video",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seg_size", type=int, default=224, help="Size of segmentation image"
    )
    parser.add_argument(
        "--blend", action="store_true", help="Toggle transparency of segmentation"
    )
    parser.add_argument("--show_bb", action="store_true", help="Show bounding boxes")
    parser.add_argument(
        "--side_by_side", action="store_true", help="Show original video to the left"
    )
    parser.add_argument("--fps", type=int, help="FPS (Frames Per Second)", default=2)
    parser.add_argument(
        "--hide_seg", action="store_true", help="Hide semantic segmentation"
    )
    parser.add_argument("pickled_file", type=str)
    parser.add_argument(
        "out_video_file", type=str, help="Out video file name. Must end with .mp4"
    )
    args = parser.parse_args()

    assert args.out_video_file.endswith(".mp4"), "Expected out mp4 file"

    return (
        args.pickled_file,
        args.out_video_file,
        args.seg_size,
        args.blend,
        args.fps,
        args.show_bb,
        args.side_by_side,
        args.hide_seg,
    )


def get_colored_seg_image(
    image_seg_arr: npt.NDArray, wanted_classes: list[int]
) -> Image.Image:
    image_seg_arr[np.isin(image_seg_arr, wanted_classes, invert=True)] = 0
    img = Image.fromarray(image_seg_arr)
    img.putpalette(colors)
    return img


def make_seg_transparent(image_seg: Image.Image, alpha: int):
    new_img_arr = np.array(image_seg)
    new_img_arr[:, :, 3] = (alpha * (new_img_arr[:, :, :3] != 0).any(axis=2)).astype(
        np.uint8
    )

    return Image.fromarray(new_img_arr)


def get_bounding_boxes_img(
    seg_image: Image.Image, wanted_classes: list[int], threshold: int = 800
) -> tuple[Image.Image, list[npt.NDArray]] | None:
    img = np.asarray(Image.new("RGB", seg_image.size))
    rects = []

    for wanted_class in wanted_classes:
        wanted_class_img = np.asarray(seg_image.convert("RGB"), dtype="uint8")
        wanted_class_img[wanted_class_img != colors[wanted_class]] = 0
        seg_arr = np.asarray(Image.fromarray(wanted_class_img).convert("L"))
        seg_arr[seg_arr != 0] = 255
        contours, _ = cv2.findContours(
            seg_arr,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        class_rects = [
            (wanted_class, cv2.boundingRect(cv2.approxPolyDP(c, 3, True)))
            for c in contours
            if cv2.contourArea(c) > threshold
        ]

        for _, rect in class_rects:
            cv2.rectangle(
                img,
                (int(rect[0]), int(rect[1])),
                (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
                (
                    int(colors[wanted_class][0]),
                    int(colors[wanted_class][1]),
                    int(colors[wanted_class][2]),
                ),
                2,
            )

        rects += class_rects

    if not rects:
        return None

    img = make_seg_transparent(Image.fromarray(img).convert("RGBA"), 255)

    return img, rects


def get_text_for_rects(combined_image: Image.Image, rects: list[npt.NDArray]):
    img = np.asarray(combined_image)

    for class_index, rect in rects:
        x, y = (rect[0] + rect[2] + 10, rect[1] + rect[3])

        # now draw the text over it
        cv2.putText(
            img,
            f"{CLASSES_NAMES[class_index]} detected",
            (x, y),
            0,
            0.5,
            (0, 255, 0, 255),
            lineType=cv2.LINE_4,
        )
    return Image.fromarray(img)


def main():
    (
        pickled_file_path,
        out_video_file,
        seg_size,
        blend,
        fps,
        show_bounding_boxes,
        side_by_side,
        hide_seg,
    ) = get_args()
    seg_res = (seg_size, seg_size)

    frames = []
    with open(pickled_file_path, "rb") as pickle_file:
        while True:
            try:
                frames += pickle.load(pickle_file)
            except (EOFError, pickle.UnpicklingError):
                break

    assert all(
        raw_image.shape
        == (PI_CAMERA_RES["height"], PI_CAMERA_RES["width"], PI_CAMERA_RES["dim"])
        and seg_image.shape == seg_res
        for raw_image, seg_image in frames
    ), f"Expected raw image to be of dimension {PI_CAMERA_RES['width']}x{PI_CAMERA_RES['height']} and segmentation image to be of dimension {seg_res[0]}x{seg_res[1]}"

    video_writer = cv2.VideoWriter(
        out_video_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (PI_CAMERA_RES["width"] * (2 if side_by_side else 1), PI_CAMERA_RES["height"]),
    )

    for raw_image_arr, seg_image in frames:
        raw_image = Image.fromarray(raw_image_arr.astype("uint8")).convert("RGBA")
        raw_seg_image = (
            get_colored_seg_image(seg_image.astype("uint8"), WANTED_CLASSES)
            .resize(
                (PI_CAMERA_RES["width"], PI_CAMERA_RES["height"]),
                resample=Image.Resampling.NEAREST,
            )
            .convert("RGBA")
        )

        combined_image = raw_image.copy()

        seg_image = (
            Image.new("RGBA", (PI_CAMERA_RES["width"], PI_CAMERA_RES["height"]))
            if hide_seg
            else make_seg_transparent(raw_seg_image, 150 if blend else 255)
        )
        if show_bounding_boxes:
            if bb_result := get_bounding_boxes_img(raw_seg_image, WANTED_CLASSES):
                bounding_boxes, rects = bb_result
                seg_image.paste(bounding_boxes, (0, 0), bounding_boxes)
            else:
                rects = None

        combined_image.paste(seg_image, (0, 0), seg_image)
        if show_bounding_boxes and rects:
            combined_image = get_text_for_rects(combined_image, rects)

        if side_by_side:
            side_by_side_image = Image.new(
                "RGBA", (PI_CAMERA_RES["width"] * 2, PI_CAMERA_RES["height"])
            )
            side_by_side_image.paste(raw_image, (0, 0))

            side_by_side_image.paste(combined_image, (PI_CAMERA_RES["width"], 0))

            video_writer.write(np.asarray(side_by_side_image.convert("RGB")))
        else:
            video_writer.write(np.asarray(combined_image.convert("RGB")))

    video_writer.release()


if __name__ == "__main__":
    main()
