#!/usr/bin/env python3

from djitellopy import Tello
import time
import numpy as np

PERSON_INDEX = 15
DISTANCE_FORMULA = 400 * np.sqrt(1200)


class DroneConnectionError(RuntimeError):
    def __init__(self, message="Couldn't connect to drone"):
        self.message = message
        super().__init__(self.message)


def try_connect(t: Tello, timeout_seconds: int):
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            t.connect()
            return
        except OSError:
            pass

    raise DroneConnectionError


def do_patrol(
    instructions: list[tuple[str, int]],
    drone_speed: int,
):
    t = Tello()

    assert all(
        hasattr(t, command) for command, _ in instructions
    ), f"All commands must exist in DJITelloPy"

    assert 0 <= drone_speed <= 100, "Expected drone speed to be between 0-100"

    try_connect(t, 60)
    time.sleep(2)
    t.set_speed(drone_speed)
    time.sleep(2)

    t.takeoff()
    time.sleep(4)

    t.send_control_command("command")
    time.sleep(2)

    for command, amount in instructions:
        getattr(t, command)(amount)
        if command in ["rotate_clockwise", "rotate_counterclockwise"]:
            time.sleep(2)

    t.land()
