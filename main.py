#!/usr/bin/env python3

from djitellopy import Tello


def main():
    t = Tello()
    t.connect()
    t.takeoff()
    t.land()


if __name__ == "__main__":
    main()
