"""Sweep regressions: run with `.venv/bin/python test_air_harp.py`."""

from types import SimpleNamespace

import air_harp as ah


class FakeAudio:
    def __init__(self):
        self.notes = []

    def play(self, note):
        self.notes.append(note)


def frame(y, pinch=False):
    """One hand centred at x=0.5 with its index tip at height y (normalised)."""
    points = [SimpleNamespace(x=0.5, y=y) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=y + 0.2)  # wrist: palm length 216 px
    points[4] = SimpleNamespace(x=0.5, y=y) if pinch else SimpleNamespace(x=0.65, y=y)
    return SimpleNamespace(hand_landmarks=[points])


def sweep(steps):
    audio = FakeAudio()
    harp = ah.Harp(audio)
    harp.add_string((500, 540), (1400, 540))  # horizontal string at y = 0.5
    for at, result in steps:
        harp.update_hands(result, at)
    return audio.notes


# Tracking drops the hand for 0.2 s mid-sweep (motion blur); the crossing must still count.
assert sweep([(0.0, frame(0.40)), (0.2, frame(0.60))]), "gap in tracking lost the pluck"

# A single-frame false pinch during the sweep must not swallow the pluck.
assert sweep([(0.0, frame(0.40)), (0.033, frame(0.47, pinch=True)), (0.066, frame(0.60))]), \
    "one-frame pinch blocked the pluck"

# A held pinch still blocks plucking (it is a drawing gesture).
assert not sweep([(0.0, frame(0.40, True)), (0.033, frame(0.47, True)), (0.066, frame(0.60, True))]), \
    "held pinch plucked"

print("ok")
