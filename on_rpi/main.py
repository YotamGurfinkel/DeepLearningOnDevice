#!/usr/bin/env python3

import driver
import exporter
import multiprocessing
import sys


def parse_instructions(file_name: str) -> list[tuple[str, int]]:
    with open(file_name) as f:
        return [(line.split()[0], int(line.split()[1])) for line in f.readlines()]


def main():
    if len(sys.argv) < 2:
        print("USAGE: python main.py NAME_OF_TRACK_FILE")
        exit(1)

    drone_instructions = parse_instructions(sys.argv[1])
    end_event = multiprocessing.Event()
    exporter_process = multiprocessing.Process(
        target=exporter.run_model_export_video_data,
        args=(end_event,),
    )

    exporter_process.start()

    try:
        driver.do_patrol(drone_instructions, 20)
    except driver.DroneConnectionError as e:
        print(e)

    end_event.set()

    exporter_process.join()


if __name__ == "__main__":
    main()
