#!/usr/bin/env python3

from djitellopy import Tello
import time
import numpy as np
import numpy.typing as npt
import multiprocessing

POSSIBLE_DIRECTIONS = ["up", "down", "left", "right", "forward", "back"]
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


def got_person_distance(person_pixels: int, distance_meters: int):
    if person_pixels == 0:
        return False

    return distance_meters - 1 / np.sqrt(person_pixels) * DISTANCE_FORMULA >= 0


def do_patrol(
    instructions: list[tuple[str, int]],
    drone_speed: int,
    biggest_person_pixels: multiprocessing.Value,
    found_bottle: multiprocessing.Value,
    new_frame_received: multiprocessing.Condition,
):
    assert all(
        direction in POSSIBLE_DIRECTIONS for direction, _ in instructions
    ), f"All directions must be in {POSSIBLE_DIRECTIONS}"

    assert 0 <= drone_speed <= 100, "Expected drone speed to be between 0-100"

    t = Tello()

    try_connect(t, 60)
    t.set_speed(drone_speed)
    time.sleep(2)

    t.takeoff()
    time.sleep(2)
    wake_drone = True
    switch_direction = False
    got_person = True
    for command, amount in instructions:
        if wake_drone:
            t.send_control_command("command")
            wake_drone = False
            time.sleep(2)

        # Wait for a new frame to arrive for each person check
        with new_frame_received:
            new_frame_received.wait()

        while got_person_distance(biggest_person_pixels.value, 200):
            got_person = True
            if switch_direction:
                t.move_right(40)
            else:
                t.move_left(40)

            with new_frame_received:
                new_frame_received.wait()

        if got_person:
            switch_direction = not switch_direction
            got_person = False

        getattr(t, command)(amount)

        if found_bottle.value:
            t.rotate_clockwise(360)
            break

    t.land()
