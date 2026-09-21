# MUST BE THE ABSOLUTE FIRST LINES OF THE FILE
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mediapipe as mp

import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sounddevice as sd
import time
import math

# hand landmark recognition model configuration
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task',
)
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# declare hand landmark recognition variables
mp_hands = vision.HandLandmarksConnections
mp_draw = vision.drawing_utils
mp_drawing_styles = vision.drawing_styles

# configure camera feed
cv2.namedWindow("Theremin", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(3, 1920)
cap.set(4, 1080)

# determine which finger segments to draw
PINCHER_LANDMARKS = [4, 3, 2, 5, 6, 7, 8]

# declare theremin control point coordinates
control_points = [(0, 0), (0, 0)]

# audio constants
SAMPLE_RATE = 44100
AUDIO_CHUNK = 256

# Global audio state
# This dictionary acts as a thread-safe container for scalar variables
audio_state = {
    'target_frequency': 440.0,  # MediaPipe updates this
    'current_frequency': 440.0, # The audio thread uses this to glide smoothly
    'target_volume': 0.5,
    'current_volume': 0.5,
    'phase': 0.0  # Accumulates the wave angle seamlessly
}

# musical notes
notes = ["A", "A#/Bb",
         "B",
         "C", "C#/Db",
         "D", "D#/Eb",
         "E",
         "F", "F#/Gb",
         "G", "G#/Ab"]

exp_c = 0.0
exp_k = 0.0

min_x = 5
max_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 5

min_pitch = 120.0
max_pitch = 1200.0

num_notes = 0
starting_note = 0
starting_note_pos = 0

# pitch + note guide initialization
def pitch_exp_setup():
    global exp_k
    global exp_c
    exp_k = math.log((max_pitch / min_pitch), 2) / (max_x - min_x)
    exp_c = math.log(min_pitch / math.pow(2, min_x * exp_k), 2) / exp_k

    global num_notes
    global starting_note_pos
    global starting_note
    num_notes = round(unrounded_midi_note(max_pitch) - unrounded_midi_note(min_pitch))
    starting_note_pos = round(get_midi_note_x_coord(unrounded_midi_note(min_pitch)))
    starting_note = round(unrounded_midi_note(min_pitch))
    starting_note = round(unrounded_midi_note(min_pitch))

def get_midi_note_x_coord(note):
    freq = math.pow(2, (note - 69) / 12) * 440
    return math.log(freq, 2) / exp_k - exp_c

def track_hands(rgb_image, results, landmarks):
    global cap
    annotated = np.copy(rgb_image)
    # 0. Draw note guide
    if control_points[0][1] < 960:
        draw_note_guide(annotated, control_points[0][1])

    for hand_landmarks in results.hand_landmarks:
        h, w, c = annotated.shape
        points = []

        # 1. Extract coordinates only for the pincher finger landmarks
        for idx in PINCHER_LANDMARKS:
            landmark = hand_landmarks[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            points.append((cx, cy))

        cpx = int((hand_landmarks[8].x + hand_landmarks[4].x) / 2 * w)
        cpy = int((hand_landmarks[8].y + hand_landmarks[4].y) / 2 * h)

        points.append((cpx, cpy))


        handedness = results.handedness[results.hand_landmarks.index(hand_landmarks)]

        if handedness[0].category_name == "Left":
            control_points[0] = (cpx,cpy)
        else:
            control_points[1] = (cpx,cpy)

        # 3. Draw lines connecting the finger joints
        for i in range(len(points) - 1):
           cv2.line(annotated, points[i], points[i + 1], (0, 0, 0), 2)

        cv2.line(annotated, points[len(points) - 2], points[len(points) - 1], (60, 255, 0), 3)
        cv2.line(annotated, points[len(points) - 1], points[0], (60, 255, 0), 3)

        # 4. Draw circles on the joint landmarks
        for i in range(len(points) - 1):
            cv2.circle(annotated, points[i], 4, (0, 0, 0), cv2.FILLED)

        # draw control point
        cv2.circle(annotated, points[len(points) - 1], 7, (0, 255, 255), cv2.FILLED)


    return annotated


# THE ZERO-LATENCY CALLBACK
def audio_callback(outdata, frames, time_info, status):
    target_freq = audio_state['target_frequency']
    curr_freq = audio_state['current_frequency']
    target_vol = audio_state['target_volume']
    vol = audio_state['current_volume']
    phase = audio_state['phase']

    # SMOOTHING FACTOR (Value between 0.001 and 0.05)
    # Lower value = smoother glide, but slightly slower response.
    # Higher value = faster response, but less smooth.
    smoothing = 0.001

    # We will build an array of frequencies, one for every single sample in this chunk
    freq_samples = np.zeros(frames)

    for i in range(frames):
        # Glides curr_freq a fraction of the way closer to target_freq on every sample
        curr_freq = curr_freq + smoothing * (target_freq - curr_freq)
        vol = vol + smoothing * (target_vol - vol)

        freq_samples[i] = curr_freq

    # Save the final current_frequency back for the next chunk
    audio_state['current_frequency'] = float(curr_freq)
    audio_state['current_volume'] = float(vol)

    # Calculate phase increments based on the gliding frequency array
    delta_phases = 2 * np.pi * freq_samples / SAMPLE_RATE

    # Accumulate the running phase sample-by-sample
    # np.cumsum calculates the rolling sum of the phase steps
    phases = phase + np.cumsum(delta_phases)

    # Save the next starting phase safely
    audio_state['phase'] = float(phases[-1] % (2 * np.pi))

    # Compute the sine wave and output
    outdata[:, 0] = vol * np.sin(phases)

def draw_note_guide(img, y_pos = 0):
    cv2.line(img, (0,y_pos), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), y_pos), (60, 255, 0), 3)
    for i in range(num_notes):
        note_center_offset = -30
        if len(notes[(starting_note + 3 + i) % 12]) == 1: note_center_offset = 0
        cv2.putText(img,
                    notes[(starting_note + 3 + i) % 12],
                    (int(get_midi_note_x_coord(starting_note + i) - 10) + note_center_offset, int(y_pos + 60 * (i % 2) - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    8,
                    cv2.LINE_AA)
        cv2.putText(img,
                    notes[(starting_note + 3 + i) % 12],
                    (int(get_midi_note_x_coord(starting_note + i) - 10) + note_center_offset, int(y_pos + 60 * (i % 2) - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)


# Start the hardware stream
stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=AUDIO_CHUNK,
    channels=1,
    callback=audio_callback
)

def clamp(n, min_val, max_val):
    return max(min_val, min(n, max_val))

def unrounded_midi_note(freq):
    return 12 * math.log(max(0.000001, freq / 440.0), 2) + 69

def main():
    global stream

    pitch_exp_setup()
    stream.start()
    print(get_midi_note_x_coord(69))

    try:
        while cap.isOpened():
            attempt = 0
            success, img = cap.read()
            while not success and attempt < 5:
                time.sleep(0.2)
                success, img = cap.read()
                attempt += 1
                if not success:
                    print(f"Failed to read frame after {attempt} attempts.")
                    break

            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)

            results = detector.detect(mp_img)

            annotated_img = track_hands(mp_img.numpy_view(), results, PINCHER_LANDMARKS)

            frequency_input = math.pow(2, (control_points[0][0] + exp_c) * exp_k)
            volume_input = 1.0 - ((clamp(control_points[1][1], 216, 864) - 216) / 648)

            audio_state['target_frequency'] = clamp(frequency_input, min_pitch, max_pitch)
            audio_state['target_volume'] = volume_input

            final_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

            volume_text = int(audio_state['current_volume'] * 100)
            pitch_text = int(audio_state['current_frequency'])

            raw_note = round(unrounded_midi_note(audio_state['current_frequency']))
            note_letter = notes[(round(raw_note) + 3) % 12]
            octave = 4 + (round(raw_note) - 60) // 12

            #display volume
            cv2.putText(final_img, f"{volume_text}%", control_points[1], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 12, cv2.LINE_AA)
            cv2.putText(final_img, f"{volume_text}%", control_points[1], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 255, 0), 2, cv2.LINE_AA)

            #display pitch
            cv2.putText(final_img, f"{pitch_text} Hz", control_points[0], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 12, cv2.LINE_AA)
            cv2.putText(final_img, f"{pitch_text} Hz", control_points[0], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 255, 0), 2, cv2.LINE_AA)

            #display closest note on 12 tone scale
            if len(note_letter) == 1:
                cv2.putText(final_img, f"{note_letter}{octave}", (control_points[0][0], control_points[0][1] - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 12, cv2.LINE_AA)
                cv2.putText(final_img, f"{note_letter}{octave}", (control_points[0][0], control_points[0][1] - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 255, 0), 2, cv2.LINE_AA)
            else:
                accidental_note_letters = f"{note_letter[:2]}{octave}{note_letter[2:]}{octave}"
                cv2.putText(final_img, accidental_note_letters, (control_points[0][0], control_points[0][1] - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 12, cv2.LINE_AA)
                cv2.putText(final_img, accidental_note_letters, (control_points[0][0], control_points[0][1] - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Theremin", final_img)

            # print(control_points)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stream.stop()
        stream.close()




if __name__ == "__main__":
    main()