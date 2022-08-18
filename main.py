#!/usr/bin/env python3

import driver
import exporter
import multiprocessing
import numpy as np
import ctypes


def main():
    end_event = multiprocessing.Event()
    biggest_person_pixels = multiprocessing.Value(ctypes.c_float, 0)
    new_frame_received = multiprocessing.Condition()
    found_bottle = multiprocessing.Value(ctypes.c_bool, False)
    exporter_process = multiprocessing.Process(
        target=exporter.run_model_export_video_data,
        args=(end_event, biggest_person_pixels, found_bottle, new_frame_received),
    )

    drone_instructions = [
        ("move_forward", 313),
        ("rotate_clockwise", 90),
        ("move_forward", 465),
        ("rotate_clockwise", 180),
        ("move_forward", 266),
        ("rotate_clockwise", 90),
        ("move_forward", 249),
        ("rotate_clockwise", 90),
        ("move_forward", 187),
        ("rotate_clockwise", 180),
        ("move_forward", 187),
    ]

    exporter_process.start()

    try:
        driver.do_patrol(
            drone_instructions,
            15,
            biggest_person_pixels,
            found_bottle,
            new_frame_received,
        )
    except driver.DroneConnectionError as e:
        print(e)

    end_event.set()

    exporter_process.join()


if __name__ == "__main__":
    main()
