#!/usr/bin/env python3

import driver
import exporter
import multiprocessing


def main():
    end_event = multiprocessing.Event()
    exporter_process = multiprocessing.Process(
        target=exporter.run_model_export_video_data,
        args=(end_event,),
    )

    drone_instructions = [
        ("move_forward", 313),
        ("rotate_clockwise", 90),
        ("move_forward", 233),
        ("move_forward", 232),
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
        driver.do_patrol(drone_instructions, 20)
    except driver.DroneConnectionError as e:
        print(e)

    end_event.set()

    exporter_process.join()


if __name__ == "__main__":
    main()
