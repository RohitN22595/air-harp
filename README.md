<div align="center">

# Air Harp

**A webcam-powered musical instrument for Windows**

Draw strings in midair · Pluck them with your fingertips · Change pitch by stretching a string

[Features](#features) · [Quick start](#quick-start) · [How to play](#how-to-play) · [Rohit and CodeSmash](#rohit-and-codesmash) · [License](#license)

</div>

---

Air Harp turns your webcam into a fullscreen instrument. Pinch with both hands and pull them apart to create a string. Sweep your index finger across it to play a note and send sparks along the string.

## Features

| Feature | What you can do |
| --- | --- |
| Two-hand tracking | See your finger joints and wrist positions over the camera feed. |
| Editable strings | Grab either endpoint with a pinch or the mouse. Hear the pitch change as the string stretches or shortens. |
| Guitar-range tuning | Play all 49 chromatic pitches from E2 to E6, including sharps. |
| Sweep preset | Play a 31-note excerpt from **“He’s a Pirate”** by sweeping down through four groups of strings. |
| Mouse mode | Draw and play strings without a webcam. |
| Fullscreen preview | See separate frame rates for the camera, hand tracking, and preview. |

## Quick start

Use 64-bit Windows, Python 3.13, and speakers or headphones. A webcam is needed for hand tracking. The first setup downloads the Python packages and the camera model.

Open PowerShell and run:

```powershell
git clone https://github.com/RohitN22595/air-harp.git
cd air-harp
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
curl.exe --fail --location --output hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
.\.venv\Scripts\python.exe air_harp.py
```

After setup, run the last command again whenever you want to open Air Harp. The MediaPipe model belongs beside `air_harp.py`. To skip webcam setup and play with your mouse, run:

```powershell
.\.venv\Scripts\python.exe air_harp.py --demo
```

## How to play

1. Keep both palms and wrists visible. Pinch your thumb and index finger together on each hand.
2. Move your hands apart to stretch a string. Release a pinch to leave it in place.
3. Sweep your index finger across the string to pluck it.
4. Pinch or click and drag either endpoint to retune. The opposite endpoint stays fixed. Shorten the string for a higher note or lengthen it for a lower note.

| Key | Action |
| --- | --- |
| **P** | Restore the Pirates layout in the available space while keeping custom strings. |
| **F** | Toggle fullscreen. |
| **Z** | Undo the last string. |
| **C** | Clear the strings. |
| **Esc** | Quit. |

### Play the Pirates preset

Air Harp opens with 31 ordered strings, including separate strings for repeated notes. Sweep downward through the middle and stop in any of the three wider gaps. Each string plays as you cross it, so your hand sets the rhythm. [Faber Music score](https://faber-product-media.s3.amazonaws.com/4ce9a0f4-f0ac-494f-b0a9-9c3cd134f13e)

## Troubleshooting

- **The webcam opens slowly or uses the wrong camera.** Try `air_harp.py --camera 1` with the same Python executable, and close other apps that are using the webcam.
- **The app sees no hands.** Keep your palm and wrist in view and light your hand from the front. The tracker uses more than the fingertip.
- **The preview and camera show different FPS.** **Cam** measures captured frames, **Track** measures hand-tracking results, and **View** measures the rendered preview. Air Harp requests 720p at up to 60 FPS. The actual capture rate depends on your webcam.
- **There is no sound.** Check your Windows output device and volume, then try plucking with the mouse in `--demo` mode.

## Rohit and CodeSmash

Air Harp is a project by **Rohit N**. [Rohit on GitHub](https://github.com/RohitN22595)

**CodeSmash** is a coding platform for DSA practice and interview preparation, with problem sets, quizzes, mock online assessments, and coding battles. [Visit CodeSmash](https://codesmash.in/)

## Contributing

Fork the repo, make a change, and open a pull request. For changes to tracking or playback, describe the gesture or control you tested and what you observed.

## License

Air Harp’s original code and documentation are available under the [MIT License](LICENSE). You may use, modify, and share them with the copyright and license notice included.

The separately downloaded MediaPipe model, third-party dependencies, and referenced music retain their own license terms and rights. [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
