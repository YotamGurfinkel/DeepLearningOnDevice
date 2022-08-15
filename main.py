#!/usr/bin/env python3

import driver
import exporter
import multiprocessing


def main():
    end_event = multiprocessing.Event()
    exporter_process = multiprocessing.Process(
        target=exporter.run_model_export_video_data, args=(end_event,)
    )

    drone_instructions = [("forward", 50)]

    exporter_process.start()

    try:
        driver.do_patrol(drone_instructions)
    except driver.DroneConnectionError as e:
        print(e)

    end_event.set()

    exporter_process.join()


if __name__ == "__main__":
    main()
