"""Draw and play an air harp in a native desktop window."""

import argparse
from collections import deque
from dataclasses import dataclass
from itertools import cycle
from math import exp, hypot, sin, tau
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, sleep

import cv2
import numpy as np
import sounddevice as sd


WIDTH, HEIGHT = 1920, 1080
WINDOWED_WIDTH, WINDOWED_HEIGHT = 960, 540
VIEW_FRAME_INTERVAL = 1 / 60
MIN_STRING = WIDTH / 16
PITCH_LENGTH_SPAN = WIDTH * 650 / 960
GRAB_RADIUS = WIDTH / 48
WINDOW = "Air Harp"
MODEL = Path(__file__).with_name("hand_landmarker.task")
PALETTE = [(205, 175, 83), (154, 107, 210), (81, 188, 207), (126, 181, 93)]
HAND_LINKS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (17, 0),
)
STRING_SAMPLES = np.linspace(0, 1, 85)
SPARK_INDEX = np.arange(14)
SPARK_ANGLE = SPARK_INDEX * 2.39996
SPARK_SPEED = 90 + SPARK_INDEX * 47 % 185
SPARK_VELOCITY = np.column_stack((np.cos(SPARK_ANGLE), np.sin(SPARK_ANGLE))) * SPARK_SPEED[:, None]
GUITAR_NOTES = range(40, 89)  # Open low E2 through fret 24 on the high E string.
PIRATE_PARTS = (
    (57, 60, 62, 62, 62, 64, 65, 65, 65, 67),
    (64, 64, 62, 60, 60, 62),
    (57, 60, 62, 62, 62, 64, 65, 65, 65, 67),
    (64, 64, 62, 60, 62),
)
PHRASE_HEIGHT = HEIGHT * 0.76
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
Point = tuple[float, float]


def note_for_length(length):
    fraction = np.clip((length - MIN_STRING) / PITCH_LENGTH_SPAN, 0, 1)
    return GUITAR_NOTES[round((1 - fraction) * (len(GUITAR_NOTES) - 1))]


def note_name(note):
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


def make_tone(note, rate):
    """A soft attack followed by harmonics that decay at different rates."""
    time = np.arange(int(rate * 1.1), dtype=np.float32) / rate
    phase = tau * (440 * 2 ** ((note - 69) / 12)) * time
    tone = (
        np.sin(phase) * np.exp(-4.3 * time)
        + 0.36 * np.sin(2 * phase) * np.exp(-8 * time)
        + 0.18 * np.sin(3 * phase) * np.exp(-12 * time)
    )
    tone *= np.minimum(time * 300, 1) * np.minimum((1.1 - time) * 25, 1)
    return np.asarray(tone * 0.23, dtype=np.float32)


class Audio:
    def __init__(self):
        self.pending = deque(maxlen=48)
        self.voices = []
        self.stream = None
        try:
            rate = int(sd.query_devices(kind="output")["default_samplerate"])
            self.tones = {note: make_tone(note, rate) for note in GUITAR_NOTES}
            for latency in (0.020, "low"):
                try:
                    self.stream = sd.OutputStream(
                        samplerate=rate, channels=1, dtype="float32", latency=latency, callback=self.mix
                    )
                    self.stream.start()
                    break
                except sd.PortAudioError:
                    self.close()
                    if latency == "low":
                        raise
        except sd.PortAudioError as error:
            self.close()
            print(f"Audio output unavailable: {error}")

    def play(self, note):
        if self.stream is not None:
            self.pending.append(self.tones[note])

    def mix(self, output, frames, _time, _status):
        output.fill(0)
        while self.pending:
            self.voices.append((self.pending.popleft(), 0))
        remaining = []
        for wave, offset in self.voices[-48:]:
            chunk = wave[offset : offset + frames]
            output[: len(chunk), 0] += chunk
            if offset + frames < len(wave):
                remaining.append((wave, offset + frames))
        self.voices = remaining
        np.clip(output, -0.9, 0.9, out=output)

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class CameraFeed:
    """Read the webcam independently and keep only its newest frame."""

    def __init__(self, capture):
        self.capture = capture
        self.on_frame = None  # Called on this thread, so tracking never waits for rendering.
        self.latest = None
        self.failed = False
        self.stop = Event()
        self.thread = Thread(target=self.read_frames, name="air-harp-camera", daemon=True)
        self.thread.start()

    def read_frames(self):
        sequence = 0
        missed = 0
        try:
            while not self.stop.is_set():
                ok, image = self.capture.read()
                if ok:
                    sequence += 1
                    captured_at = monotonic()
                    self.latest = (sequence, captured_at, image)
                    missed = 0
                    if self.on_frame is not None:
                        self.on_frame(image, captured_at)
                elif missed >= 9:
                    break
                else:
                    missed += 1
                    self.stop.wait(0.03)
        finally:
            self.capture.release()
            self.failed = not self.stop.is_set()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.75)


def crossing(start, end, a, b):
    """Return the intersection fractions along the swipe and along the string."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    sx, sy = b[0] - a[0], b[1] - a[1]
    denominator = dx * sy - dy * sx
    if abs(denominator) < 0.001:
        return None
    ax, ay = a[0] - start[0], a[1] - start[1]
    travel = (ax * sy - ay * sx) / denominator
    contact = (ax * dy - ay * dx) / denominator
    if 0 < travel <= 1 and 0 <= contact <= 1:
        return travel, contact
    return None


@dataclass
class String:
    a: Point
    b: Point
    note: int
    color: tuple[int, int, int]
    struck_at: float = float("-inf")
    contact: float = 0.5
    direction: int = 1
    phrase_step: int | None = None

    def set_endpoint(self, end, point):
        if end == 0:
            self.a = point
        else:
            self.b = point
        self.note = note_for_length(distance_between(self.a, self.b))

    def strike(self, now, audio, contact=0.5, direction=1):
        # The cooldown only stops jitter re-hitting in the same direction; a strum
        # coming back the other way always sounds.
        if direction != self.direction or now - self.struck_at >= 0.14:
            self.struck_at = now
            self.contact = contact
            self.direction = direction
            audio.play(self.note)


@dataclass
class Hand:
    points: tuple[Point, ...]
    pinched: bool
    raw_pinch: bool = False  # This frame alone looks pinched; confirmed on the next one.

    @property
    def tip(self):
        return self.points[8]


@dataclass
class EndpointDrag:
    string: String
    end: int
    original: Point
    offset: Point
    hand: int | None
    last_seen: float


class Harp:
    def __init__(self, audio):
        self.audio = audio
        self.strings = deque(maxlen=48)
        self.colors = cycle(PALETTE)
        self.hands = {}
        self.preview = None
        self.mouse = None
        self.last_frame_at = None
        self.recent_hands = {}
        self.drag = None
        self.auditioned_note = None
        self.auditioned_at = float("-inf")

    def add_string(self, a, b, now=None, *, phrase_step=None):
        length = hypot(b[0] - a[0], b[1] - a[1])
        if length >= MIN_STRING:
            string = String(a, b, note_for_length(length), next(self.colors),
                            phrase_step=phrase_step)
            self.strings.append(string)
            if now is not None:
                string.strike(now, self.audio)

    def load_pirate_notes(self):
        custom = [string for string in self.strings if string.phrase_step is None]
        self.strings.clear()
        self.strings.extend(custom)
        gap_rows = 2
        row_count = sum(map(len, PIRATE_PARTS)) + (len(PIRATE_PARTS) - 1) * gap_rows
        row_spacing = PHRASE_HEIGHT / (row_count - 1)
        first_y = (HEIGHT - PHRASE_HEIGHT) / 2
        step = 0
        for part, notes in enumerate(PIRATE_PARTS):
            for note in notes:
                if len(self.strings) == self.strings.maxlen:
                    return
                fraction = (GUITAR_NOTES[-1] - note) / (len(GUITAR_NOTES) - 1)
                length = MIN_STRING + fraction * PITCH_LENGTH_SPAN
                y = first_y + row_spacing * (step + part * gap_rows)
                self.add_string(((WIDTH - length) / 2, y), ((WIDTH + length) / 2, y),
                                phrase_step=step)
                step += 1

    def pluck(self, start, end, now):
        distance = hypot(end[0] - start[0], end[1] - start[1])
        if distance > PHRASE_HEIGHT + 60:
            return
        hits = []
        for string in self.strings:
            if string.phrase_step is None and distance > WIDTH * 0.45:
                continue
            intersection = crossing(start, end, string.a, string.b)
            if intersection is None:
                continue
            travel, contact = intersection
            dx, dy = string.b[0] - string.a[0], string.b[1] - string.a[1]
            side = dx * (end[1] - start[1]) - dy * (end[0] - start[0])
            direction = 1 if side > 0 else -1
            hits.append((travel, string, contact, direction))
        for _, string, contact, direction in sorted(hits, key=lambda hit: hit[0]):
            string.strike(now, self.audio, contact, direction)

    def audition_drag(self, now):
        if self.drag is not None:
            note = self.drag.string.note
        else:
            length = distance_between(*self.preview) if self.preview is not None else 0
            if length < MIN_STRING:
                self.auditioned_note = None
                return
            note = note_for_length(length)
        if note != self.auditioned_note and now - self.auditioned_at >= 0.075:
            self.audio.play(note)
            self.auditioned_note = note
            self.auditioned_at = now

    def grab_endpoint(self, point, now, hand=None):
        nearest = None
        best_distance = GRAB_RADIUS
        for string in reversed(self.strings):
            for end, anchor in enumerate((string.a, string.b)):
                distance = distance_between(point, anchor)
                if distance < best_distance:
                    nearest = (string, end, anchor)
                    best_distance = distance
        if nearest is None:
            return False
        string, end, anchor = nearest
        offset = (anchor[0] - point[0], anchor[1] - point[1])
        self.drag = EndpointDrag(string, end, anchor, offset, hand, now)
        return True

    def move_endpoint(self, drag, point):
        fixed = drag.string.b if drag.end == 0 else drag.string.a
        x = min(WIDTH - 1, max(0, point[0] + drag.offset[0]))
        y = min(HEIGHT - 1, max(0, point[1] + drag.offset[1]))
        dx, dy = x - fixed[0], y - fixed[1]
        distance = hypot(dx, dy)
        if 0 < distance < MIN_STRING:
            scale = MIN_STRING / distance
            x, y = fixed[0] + dx * scale, fixed[1] + dy * scale
        if distance > 0 and 0 <= x < WIDTH and 0 <= y < HEIGHT:
            drag.string.set_endpoint(drag.end, (x, y))

    def finish_drag(self, now, commit):
        drag = self.drag
        self.drag = None
        if not commit:
            drag.string.set_endpoint(drag.end, drag.original)
            return
        endpoint = drag.string.a if drag.end == 0 else drag.string.b
        if distance_between(endpoint, drag.original) > 1:
            drag.string.struck_at = float("-inf")
            drag.string.strike(now, self.audio, contact=0.12 if drag.end == 0 else 0.88)

    def update_hands(self, result, now, viewport=(0, 0, WIDTH, HEIGHT)):
        if self.last_frame_at is not None and now - self.last_frame_at > 0.25:
            self.hands, self.preview = {}, None
        self.last_frame_at = now
        x, y, width, height = viewport
        detections = [
            tuple((x + landmark.x * width, y + landmark.y * height) for landmark in landmarks)
            for landmarks in result.hand_landmarks
        ]
        # Keep wrist identities through brief missed frames without drawing stale joints.
        # Fast sweeps blur the hand out for a few frames; the bridge keeps those plucks.
        self.recent_hands = {
            slot: (seen, hand) for slot, (seen, hand) in self.recent_hands.items()
            if now - seen <= 0.3
        }
        recent = {slot: hand for slot, (_, hand) in self.recent_hands.items()}
        slots = assign_hands(detections, recent)
        hands = {}
        for slot, points in zip(slots, detections):
            previous = recent.get(slot)
            palm = distance_between(points[0], points[9])
            gap = distance_between(points[8], points[4])
            threshold = 0.44 if previous and previous.pinched else 0.32
            if previous:
                motion = distance_between(points[8], previous.tip)
                weight = min(1, 0.66 + motion / (WIDTH * 0.07))
                points = tuple(
                    (px + (x - px) * weight, py + (y - py) * weight)
                    for (px, py), (x, y) in zip(previous.points, points)
                )
            raw = palm > 0 and gap < palm * threshold
            # One pinched-looking frame mid-sweep must not swallow a pluck.
            confirmed = raw and previous is not None and (previous.pinched or previous.raw_pinch)
            hands[slot] = Hand(points, confirmed, raw)

        self.update_gestures(hands, now, recent)
        self.hands = hands
        for slot, hand in hands.items():
            self.recent_hands[slot] = (now, hand)

    def update_gestures(self, hands, now, recent):
        if self.drag is not None:
            drag = self.drag
            if drag.hand is None:
                return
            hand = hands.get(drag.hand)
            if now - drag.last_seen > 0.22:
                self.finish_drag(now, commit=False)
            elif hand is not None:
                if hand.pinched:
                    self.move_endpoint(drag, hand.tip)
                    drag.last_seen = now
                else:
                    self.finish_drag(now, commit=drag.hand in self.hands)
            return

        if self.preview is None:
            for slot, hand in hands.items():
                previous = self.hands.get(slot)
                if hand.pinched and (previous is None or not previous.pinched):
                    if self.grab_endpoint(hand.tip, now, hand=slot):
                        return

        if len(hands) == 2 and all(hand.pinched for hand in hands.values()):
            self.preview = (hands[0].tip, hands[1].tip)
        elif self.preview is not None:
            # A visible opening gesture commits the string. Lost tracking cancels it.
            if any(self.hands[slot].pinched and not hand.pinched
                   for slot, hand in hands.items() if slot in self.hands):
                self.add_string(*self.preview, now)
            self.preview = None

        for slot, hand in hands.items():
            previous = recent.get(slot)
            # Drawing (preview) and endpoint drags already return above; a curled
            # strumming hand often reads as pinched, so pinch alone must not block.
            if previous and self.preview is None:
                self.pluck(previous.tip, hand.tip, now)

    def expire_tracking(self, now):
        if self.last_frame_at is not None and now - self.last_frame_at > 0.25:
            self.reset_tracking(now)

    def reset_tracking(self, now):
        self.last_frame_at = None
        self.hands, self.preview = {}, None
        self.recent_hands.clear()
        if self.drag is not None and self.drag.hand is not None:
            self.finish_drag(now, commit=False)

    def mouse_event(self, event, x, y, flags, demo):
        point = (x, y)
        now = monotonic()
        if self.drag is not None:
            if self.drag.hand is None:
                if event == cv2.EVENT_LBUTTONUP:
                    self.move_endpoint(self.drag, point)
                    self.finish_drag(now, commit=True)
                elif event == cv2.EVENT_MOUSEMOVE:
                    if flags & cv2.EVENT_FLAG_LBUTTON:
                        self.move_endpoint(self.drag, point)
                    else:
                        self.finish_drag(now, commit=True)
        elif event == cv2.EVENT_LBUTTONDOWN and self.preview is None:
            if not self.grab_endpoint(point, now) and demo:
                self.preview = (point, point)
        elif demo and event == cv2.EVENT_LBUTTONUP and self.preview is not None:
            self.add_string(self.preview[0], point, now)
            self.preview = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.preview is not None:
                self.preview = (self.preview[0], point)
            elif self.mouse is not None:
                self.pluck(self.mouse, point, now)
        self.mouse = point


def pixel(point):
    return tuple(round(coord) for coord in point)


def distance_between(a, b):
    return hypot(b[0] - a[0], b[1] - a[1])


def assign_hands(detections, previous):
    if not detections:
        return ()

    def cost(slots):
        return sum(
            distance_between(points[0], previous[slot].points[0])
            for slot, points in zip(slots, detections) if slot in previous
        )

    if len(detections) == 2:
        return min(((0, 1), (1, 0)), key=cost)
    if previous:
        return (min(previous, key=lambda slot: cost((slot,))),)
    return (0,)

def draw_string(frame, a, b, color, now, age=None, contact=0.5, direction=1,
                active_end=None):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length = hypot(dx, dy)
    if length < 1:
        return
    normal = np.array([-dy, dx]) / length
    travel_rate = min(3.1, 1350 / length)
    spark_scale = min(1.35, 0.62 + length / (WIDTH * 0.65))
    t = STRING_SAMPLES
    envelope = np.sin(np.pi * t)
    if age is None:
        # The preview flexes as the string is held under tension.
        shape = min(10, length * 0.012) * np.sin(tau * (3 * t - now * 2.6)) * envelope
    elif age < 0.9:
        # A localized bend breaks into two waves and a damped standing vibration.
        advance = age * travel_rate
        left = np.exp(-((t - contact + advance) / 0.095) ** 2)
        right = np.exp(-((t - contact - advance) / 0.095) ** 2)
        pulse = (left + right) * 0.5 * exp(-5 * age)
        ringing = (
            0.50 * sin(tau * 8 * age) * envelope
            + 0.24 * sin(tau * 13 * age) * np.sin(2 * np.pi * t)
        ) * exp(-4 * age)
        shape = direction * min(49, length * 0.16) * (pulse * envelope + ringing)
    else:
        shape = 0.65 * np.sin(tau * (2 * t - now * 0.85)) * envelope

    points = np.array([ax, ay]) + t[:, None] * np.array([dx, dy]) + shape[:, None] * normal
    # Subpixel positions keep the thin string smooth and visible when upscaled.
    line = np.rint(points * 16).astype(np.int32)
    cv2.polylines(frame, [line], False, color, 2, cv2.LINE_AA, shift=4)
    for index, end in enumerate((a, b)):
        size = 6 if age is None else 4
        if index == active_end:
            size = 9
        draw_star(frame, end, color, size)

    if age is None:
        for offset in (0.12, 0.52):
            position = (offset + now * travel_rate * 0.35) % 1
            draw_string_pulse(frame, points, position, 1, color, 0.75, spark_scale)
    elif 0 <= age < 0.6:
        for side in (-1, 1):
            position = contact + side * age * travel_rate
            if 0 <= position <= 1:
                draw_string_pulse(frame, points, position, side, color,
                                  1 - age / 0.6, spark_scale)
        draw_spark_burst(frame, points[round(contact * (len(points) - 1))],
                         normal, color, age, direction, spark_scale)


def draw_star(frame, point, color, radius):
    x, y = pixel(point)
    cv2.line(frame, (x - radius, y), (x + radius, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - radius - 2), (x, y + radius + 2), color, 1, cv2.LINE_AA)


def draw_string_pulse(frame, points, position, direction, color, intensity, scale):
    samples = len(points) - 1
    head = round(position * samples)
    tail = min(samples, max(0, head - direction * round(7 * scale)))
    segment = points[min(head, tail):max(head, tail) + 1]
    core = tuple(round(channel + (255 - channel) * intensity * 0.9) for channel in color)
    if len(segment) > 1:
        stroke = np.rint(segment).astype(np.int32)
        cv2.polylines(frame, [stroke], False, color, 4, cv2.LINE_AA)
        cv2.polylines(frame, [stroke], False, core, 1, cv2.LINE_AA)
    draw_star(frame, points[head], core, max(2, round(7 * scale * intensity)))


def draw_spark_burst(frame, origin, normal, color, age, direction, scale):
    velocity = scale * (SPARK_VELOCITY + direction * normal * 60)
    elapsed = np.maximum(0, age - SPARK_INDEX % 4 * 0.018)
    gravity = np.array([0, 330])
    heads = origin + velocity * elapsed[:, None] + gravity * elapsed[:, None] ** 2
    tail_time = np.maximum(0, elapsed - 0.032)
    tails = origin + velocity * tail_time[:, None] + gravity * tail_time[:, None] ** 2
    visibility = np.clip(1 - elapsed / (0.35 + SPARK_INDEX % 5 * 0.045), 0, 1)
    for head, tail, life, time in zip(heads, tails, visibility, elapsed):
        if time > 0 and life > 0:
            tint = tuple(round((channel * 0.35 + 255 * 0.65) * (0.4 + 0.6 * life))
                         for channel in color)
            cv2.line(frame, pixel(tail), pixel(head), tint, 1, cv2.LINE_AA)
            if life > 0.45:
                cv2.circle(frame, pixel(head), 1, (216, 244, 255), -1, cv2.LINE_AA)


def draw_hand(frame, hand):
    points = tuple(map(pixel, hand.points))
    for a, b in HAND_LINKS:
        cv2.line(frame, points[a], points[b], (38, 181, 53), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, (46, 54, 221), -1, cv2.LINE_AA)
    if hand.pinched:
        cv2.circle(frame, points[8], 8, (226, 240, 246), 1, cv2.LINE_AA)


def write(frame, words, origin, size=0.52, color=(226, 233, 245)):
    cv2.putText(frame, words, origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def render(frame, harp, now, demo, rates=""):
    # The tracking thread mutates the harp; draw from one snapshot of it.
    strings, hands = list(harp.strings), list(harp.hands.values())
    drag, preview = harp.drag, harp.preview
    for string in strings:
        active_end = drag.end if drag and drag.string is string else None
        age = now - string.struck_at if active_end is None else None
        draw_string(frame, string.a, string.b, string.color, now, age,
                    string.contact, string.direction, active_end)
    if preview is not None:
        a, b = preview
        draw_string(frame, a, b, (186, 191, 130), now)

    for hand in hands:
        draw_hand(frame, hand)
    for string in strings:
        editing_b = drag and drag.string is string and drag.end == 1
        draw_note(frame, string.b if editing_b else string.a, string.note,
                  beside_endpoint=string.phrase_step is not None and not editing_b)
    if preview is not None:
        length = distance_between(*preview)
        if length >= MIN_STRING:
            draw_note(frame, preview[1], note_for_length(length))

    top, bottom = frame[:88], frame[-60:]
    cv2.convertScaleAbs(top, dst=top, alpha=0.42)
    cv2.convertScaleAbs(bottom, dst=bottom, alpha=0.42)
    write(frame, "AIR HARP", (36, 54), 1.2, (255, 233, 168))
    guide = ("Drag to draw  /  Sweep to pluck  /  Drag either end to retune" if demo else
             "Pinch both hands to draw  /  Sweep to pluck  /  Pinch or drag an end to retune")
    show_phrase = any(string.phrase_step is not None for string in harp.strings)
    write(frame, guide, (260, 38 if show_phrase else 53), 0.76)
    if show_phrase:
        write(frame, "He's a Pirate   /   Sweep down the middle and stop in any gap",
              (260, 70), 0.68, (255, 233, 168))
    tracking = "Mouse mode" if demo else f"Hands {len(hands)}/2"
    if not demo and not hands:
        tracking += "  /  Show your palm and wrist"
    write_right(frame, tracking, 37 if rates else 53)
    if rates:
        write_right(frame, rates, 69)
    write(frame, "P  Pirate strings     F  Fullscreen     Z  Undo     C  Clear     Esc  Quit",
          (36, HEIGHT - 22), 0.73)
    sound = "" if harp.audio.stream is not None else "  /  Audio unavailable"
    write_right(frame, f"E2-E6 chromatic  /  {len(strings)} strings{sound}", HEIGHT - 22)


def draw_note(frame, anchor, note, *, beside_endpoint=False):
    name = note_name(note)
    width, height = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)[0]
    x = anchor[0] - width - 17 if beside_endpoint else anchor[0] - width / 2
    y = anchor[1] + height / 2 if beside_endpoint else anchor[1] - 14
    x = round(min(WIDTH - width - 8, max(8, x)))
    y = round(min(HEIGHT - 68, max(112, y)))
    cv2.rectangle(frame, (x - 6, y - height - 4), (x + width + 6, y + 5),
                  (24, 23, 24), -1)
    write(frame, name, (x, y), 0.60)


def write_right(frame, words, y):
    width, _ = cv2.getTextSize(words, cv2.FONT_HERSHEY_SIMPLEX, 0.73, 1)[0]
    write(frame, words, (WIDTH - width - 36, y), 0.73)


def fit_camera_view(width, height):
    scale = min(WIDTH / width, HEIGHT / height)
    fitted_width, fitted_height = round(width * scale), round(height * scale)
    return ((WIDTH - fitted_width) // 2, (HEIGHT - fitted_height) // 2,
            fitted_width, fitted_height)


def camera_preview(image, viewport):
    x, y, width, height = viewport
    preview = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if width == WIDTH and height == HEIGHT:
        return preview
    frame = np.full((HEIGHT, WIDTH, 3), (24, 23, 24), dtype=np.uint8)
    frame[y:y + height, x:x + width] = preview
    return frame


def create_tracker(handle):
    """Run hand tracking; `handle(captured_at, result)` fires on MediaPipe's thread."""
    import mediapipe as mp

    if not MODEL.is_file():
        raise RuntimeError(f"Missing hand tracking model: {MODEL}. See README.md for setup.")

    def on_result(result, _image, timestamp):
        handle(timestamp / 1000, result)

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        result_callback=on_result,
    )
    return vision.HandLandmarker.create_from_options(options), mp


def frame_tracker(detector, mp, shared):
    """Feed each camera frame to the detector straight from the camera thread."""
    last_timestamp = -1

    def track(image, captured_at):
        nonlocal last_timestamp
        height, width = image.shape[:2]
        shared["viewport"] = fit_camera_view(width, height)
        scale = min(1, 640 / max(width, height))
        small = cv2.resize(image, (round(width * scale), round(height * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1 else image
        rgb = cv2.cvtColor(cv2.flip(small, 1), cv2.COLOR_BGR2RGB)
        last_timestamp = max(last_timestamp + 1, int(captured_at * 1000))
        detector.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                              last_timestamp)

    return track


def open_camera(index, fast_mode=True):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        return None
    if fast_mode:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not camera.set(cv2.CAP_PROP_FPS, 60):
            camera.set(cv2.CAP_PROP_FPS, 30)
    return CameraFeed(camera)


def run(harp, camera_index, demo, fullscreen=True, log_path=None):
    feed = None
    detector = None
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(WINDOW, WINDOWED_WIDTH, WINDOWED_HEIGHT)
        cv2.imshow(WINDOW, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if not demo:
            feed = open_camera(camera_index)
            if feed is None:
                demo = True
                print(f"Webcam {camera_index} unavailable. Starting mouse mode.")
        # Plucks fire on MediaPipe's thread the moment a result lands; the lock keeps
        # them from interleaving with keyboard, mouse and timeout updates.
        lock = Lock()
        shared = {"viewport": (0, 0, WIDTH, HEIGHT), "tracked": 0, "demo": demo}

        log = open(log_path, "w", buffering=1) if log_path else None
        if log:
            log.write("captured,handled,hands,raw_detections,tip_x,tip_y,pinched,strikes\n")

        def on_tracked(captured_at, result):
            with lock:
                shared["tracked"] += 1
                if not shared["demo"]:
                    harp.update_hands(result, captured_at, shared["viewport"])
                if log:
                    hand = harp.hands.get(0) or next(iter(harp.hands.values()), None)
                    tip = hand.tip if hand else ("", "")
                    strikes = sum(s.struck_at == captured_at for s in harp.strings)
                    log.write(f"{captured_at:.4f},{monotonic():.4f},{len(harp.hands)},"
                              f"{len(result.hand_landmarks)},{tip[0]:.0f},{tip[1]:.0f},"
                              f"{int(bool(hand and hand.pinched))},{strikes}\n"
                              if hand else
                              f"{captured_at:.4f},{monotonic():.4f},0,"
                              f"{len(result.hand_landmarks)},,,0,0\n")

        def on_mouse(event, x, y, flags, mouse_demo):
            with lock:
                harp.mouse_event(event, x, y, flags, mouse_demo)

        if feed is not None:
            detector, mp = create_tracker(on_tracked)
            feed.on_frame = frame_tracker(detector, mp, shared)

        cv2.setMouseCallback(WINDOW, on_mouse, demo)

        background = np.full((HEIGHT, WIDTH, 3), (39, 24, 22), dtype=np.uint8)
        viewport = (0, 0, WIDTH, HEIGHT)
        last_sequence = None
        measured_at = monotonic()
        camera_frames = tracking_frames = view_frames = 0
        rates = ""
        tried_default_camera_mode = False
        while True:
            frame_started = monotonic()
            if feed is not None and feed.failed:
                retry_default = feed.latest is None and not tried_default_camera_mode
                feed.close()
                feed = open_camera(camera_index, fast_mode=False) if retry_default else None
                tried_default_camera_mode = True
                if feed is not None:
                    feed.on_frame = frame_tracker(detector, mp, shared)
                else:
                    print(f"Webcam {camera_index} stopped supplying frames. Starting mouse mode.")
                    detector.close()
                    detector = None
                    with lock:
                        shared["demo"] = demo = True
                        harp.reset_tracking(monotonic())
                    background.fill(24)
                    rates = ""
                    cv2.setMouseCallback(WINDOW, on_mouse, True)

            latest = feed.latest if feed is not None else None
            if latest is not None and latest[0] != last_sequence:
                sequence, captured_at, image = latest
                if last_sequence is not None:
                    camera_frames += sequence - last_sequence
                last_sequence = sequence
                image = cv2.flip(image, 1)
                height, width = image.shape[:2]
                viewport = fit_camera_view(width, height)
                background = camera_preview(image, viewport)

            now = monotonic()
            with lock:
                tracking_frames += shared["tracked"]
                shared["tracked"] = 0
                harp.expire_tracking(now)
                harp.audition_drag(now)
            if now - measured_at >= 1:
                elapsed = now - measured_at
                rates = (f"Cam {camera_frames / elapsed:.0f}  Track {tracking_frames / elapsed:.0f}  "
                         f"View {view_frames / elapsed:.0f} fps") if not demo else ""
                if log and rates:
                    log.write(f"# {now:.4f} {rates}\n")
                measured_at = now
                camera_frames = tracking_frames = view_frames = 0
            frame = background.copy()
            render(frame, harp, now, demo, rates)
            cv2.imshow(WINDOW, frame)
            view_frames += 1
            key = cv2.pollKey() & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("f"), ord("F")):
                if fullscreen:
                    x, y, width, height = cv2.getWindowImageRect(WINDOW)
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW, WINDOWED_WIDTH, WINDOWED_HEIGHT)
                    cv2.moveWindow(WINDOW, x + max(0, (width - WINDOWED_WIDTH) // 2),
                                   y + max(0, (height - WINDOWED_HEIGHT) // 2))
                else:
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                fullscreen = not fullscreen
            with lock:
                if key in (ord("c"), ord("C")):
                    if harp.drag is not None:
                        harp.finish_drag(now, commit=False)
                    harp.strings.clear()
                if key in (ord("z"), ord("Z")) and harp.strings:
                    if harp.drag is not None:
                        harp.finish_drag(now, commit=False)
                    harp.strings.pop()
                if key in (ord("p"), ord("P")) and harp.drag is None:
                    harp.load_pirate_notes()
            remaining = VIEW_FRAME_INTERVAL - (monotonic() - frame_started)
            if remaining > 0:
                sleep(remaining)
    finally:
        # Stop the camera thread first: it is the one feeding the detector.
        if feed is not None:
            feed.close()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=0, type=int, help="Webcam index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Play with the mouse without a webcam")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Start fullscreen (default: resizable window)")
    parser.add_argument("--log", help="Write one CSV row per tracking result to this file")
    args = parser.parse_args()
    audio = Audio()
    try:
        harp = Harp(audio)
        harp.load_pirate_notes()
        run(harp, args.camera, args.demo, fullscreen=args.fullscreen, log_path=args.log)
    except (RuntimeError, cv2.error) as error:
        parser.exit(1, f"Air Harp: {error}\n")
    except KeyboardInterrupt:
        pass
    finally:
        audio.close()


if __name__ == "__main__":
    main()
