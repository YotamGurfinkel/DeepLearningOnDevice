#!/usr/bin/env python3

from djitellopy import Tello
import time

POSSIBLE_DIRECTIONS = ["up", "down", "left", "right", "forward", "back"]


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


def do_patrol(instructions: list[tuple[str, int]]):
    assert all(
        direction in POSSIBLE_DIRECTIONS for direction, _ in instructions
    ), f"All directions must be in {POSSIBLE_DIRECTIONS}"

    t = Tello()

    try_connect(t, 60)

    t.takeoff()
    for instruction in instructions:
        t.move(*instruction)

    t.land()
