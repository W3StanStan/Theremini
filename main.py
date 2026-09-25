# MUST BE THE ABSOLUTE FIRST LINES OF THE FILE
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mediapipe as mp

import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sounddevice as sd
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

# global audio state
audio_state = {
    'target_frequency': 440.0,  # MediaPipe updates this
    'current_frequency': 440.0, # audio thread uses this to glide smoothly
    'target_volume': 0.5,
    'current_volume': 0.5,
    'phase': 0.0  # keeps track of wave angle
}

# musical notes
NOTES = ["A", "A#/Bb",
         "B",
         "C", "C#/Db",
         "D", "D#/Eb",
         "E",
         "F", "F#/Gb",
         "G", "G#/Ab"]

# initialize constants for converting control point coords to frequency
exp_c = 0.0
exp_k = 0.0

# min and max right hand x coord to use for frequency
MIN_RH_X = 5
MAX_RH_X = cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 5

# min and max pitch
MIN_PITCH = 120.0
MAX_PITCH = 1200.0

# initialization of note guide helper variables
num_notes = 0
starting_note = 0
starting_note_pos = 0

# pitch + note guide initialization
def pitch_exp_setup():
    global exp_k
    global exp_c
    exp_k = math.log((MAX_PITCH / MIN_PITCH), 2) / (MAX_RH_X - MIN_RH_X)
    exp_c = math.log(MIN_PITCH / math.pow(2, MIN_RH_X * exp_k), 2) / exp_k

    global num_notes
    global starting_note_pos
    global starting_note
    num_notes = round(unrounded_midi_note(MAX_PITCH) - unrounded_midi_note(MIN_PITCH))
    starting_note_pos = round(get_midi_note_x_coord(unrounded_midi_note(MIN_PITCH)))
    starting_note = round(unrounded_midi_note(MIN_PITCH))
    starting_note = round(unrounded_midi_note(MIN_PITCH))

# convert midi note to x coordinate (A4 = 69)
def get_midi_note_x_coord(note):
    freq = math.pow(2, (note - 69) / 12) * 440
    return math.log(freq, 2) / exp_k - exp_c

# track and draw indicators on hands
def track_hands(rgb_image, results, landmarks):
    global cap
    annotated = np.copy(rgb_image)
    # draw note guide
    if control_points[0][1] < 960:
        draw_note_guide(annotated, control_points[0][1])

    # hand landmarks detected consist of nodes, iterate through them
    for hand_landmarks in results.hand_landmarks:
        h, w, c = annotated.shape
        points = []

        # extract coordinates only for the pincher finger landmarks
        for idx in PINCHER_LANDMARKS:
            landmark = hand_landmarks[idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            points.append((cx, cy))

        # calculate coords and display control point
        cpx = int((hand_landmarks[8].x + hand_landmarks[4].x) / 2 * w)
        cpy = int((hand_landmarks[8].y + hand_landmarks[4].y) / 2 * h)

        points.append((cpx, cpy))

        # differentiate handedness for control point assignment
        handedness = results.handedness[results.hand_landmarks.index(hand_landmarks)]

        if handedness[0].category_name == "Left":
            control_points[0] = (cpx,cpy)
        else:
            control_points[1] = (cpx,cpy)

        # draw lines connecting the finger nodes
        for i in range(len(points) - 1):
           cv2.line(annotated, points[i], points[i + 1], (0, 0, 0), 2)

        cv2.line(annotated, points[len(points) - 2], points[len(points) - 1], (60, 255, 0), 3)
        cv2.line(annotated, points[len(points) - 1], points[0], (60, 255, 0), 3)

        # draw circles on the finger nodes
        for i in range(len(points) - 1):
            cv2.circle(annotated, points[i], 4, (0, 0, 0), cv2.FILLED)

        # draw control point
        cv2.circle(annotated, points[len(points) - 1], 7, (0, 255, 255), cv2.FILLED)

    return annotated


# sounddevice audio stream async callback
def audio_callback(outdata, frames, time_info, status):
    target_freq = audio_state['target_frequency']
    curr_freq = audio_state['current_frequency']
    target_vol = audio_state['target_volume']
    vol = audio_state['current_volume']
    phase = audio_state['phase']

    # smoothing value
    # lower value makes for smoother glide but slightly slower response
    # higher value makes for faster response but less smooth
    smoothing = 0.001

    # build array of frequencies, one for every single sample in this chunk
    freq_samples = np.zeros(frames)

    for i in range(frames):
        # glide curr_freq a fraction of the way closer to target_freq on every sample
        curr_freq = curr_freq + smoothing * (target_freq - curr_freq)
        vol = vol + smoothing * (target_vol - vol)

        freq_samples[i] = curr_freq

    # save the final current_frequency back for the next chunk
    audio_state['current_frequency'] = float(curr_freq)
    audio_state['current_volume'] = float(vol)

    # calculate phase increments based on the gliding frequency array
    delta_phases = 2 * np.pi * freq_samples / SAMPLE_RATE

    # accumulate the running phase sample-by-sample
    phases = phase + np.cumsum(delta_phases)

    # save the next starting phase
    audio_state['phase'] = float(phases[-1] % (2 * np.pi))

    # compute sine wave and output
    outdata[:, 0] = vol * np.sin(phases)

# draw guide line with notes at their respective positions
def draw_note_guide(img, y_pos = 0):
    cv2.line(img, (0,y_pos), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), y_pos), (60, 255, 0), 3)
    for i in range(num_notes):
        note_center_offset = -30
        if len(NOTES[(starting_note + 3 + i) % 12]) == 1: note_center_offset = 0
        cv2.putText(img,
                    NOTES[(starting_note + 3 + i) % 12],
                    (int(get_midi_note_x_coord(starting_note + i) - 10) + note_center_offset, int(y_pos + 60 * (i % 2) - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    8,
                    cv2.LINE_AA)
        cv2.putText(img,
                    NOTES[(starting_note + 3 + i) % 12],
                    (int(get_midi_note_x_coord(starting_note + i) - 10) + note_center_offset, int(y_pos + 60 * (i % 2) - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)


# Start the audio stream
stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=AUDIO_CHUNK,
    channels=1,
    callback=audio_callback
)

def clamp(n, min_val, max_val):
    return max(min_val, min(n, max_val))

# convert freq to midi note
def unrounded_midi_note(freq):
    return 12 * math.log(max(0.000001, freq / 440.0), 2) + 69

def main():

    #initialization
    global stream
    pitch_exp_setup()
    stream.start()

    try:
        while cap.isOpened():
            
            success, img = cap.read()
            if not success or img is None:
                print("Ignoring empty camera frame")
                continue

            # flip video feed
            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # convert image and run detection
            mp_img = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)
            results = detector.detect(mp_img)
            annotated_img = track_hands(mp_img.numpy_view(), results, PINCHER_LANDMARKS)

            # normalize pitch change over pixel distance
            frequency_input = math.pow(2, (control_points[0][0] + exp_c) * exp_k)
            volume_input = 1.0 - ((clamp(control_points[1][1], 216, 864) - 216) / 648)

            audio_state['target_frequency'] = clamp(frequency_input, MIN_PITCH, MAX_PITCH)
            audio_state['target_volume'] = volume_input

            # convert processed feed back to displayable format
            final_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

            volume_text = int(audio_state['current_volume'] * 100)
            pitch_text = int(audio_state['current_frequency'])

            # calculations for current note display
            raw_note = round(unrounded_midi_note(audio_state['current_frequency']))
            note_letter = NOTES[(round(raw_note) + 3) % 12]
            octave = 4 + (round(raw_note) - 60) // 12

            # display volume
            cv2.putText(final_img, f"{volume_text}%", control_points[1], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 12, cv2.LINE_AA)
            cv2.putText(final_img, f"{volume_text}%", control_points[1], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 255, 0), 2, cv2.LINE_AA)

            # display pitch
            cv2.putText(final_img, f"{pitch_text} Hz", control_points[0], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 12, cv2.LINE_AA)
            cv2.putText(final_img, f"{pitch_text} Hz", control_points[0], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 255, 0), 2, cv2.LINE_AA)

            # display closest note on 12 tone scale
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

            # display final image
            cv2.imshow("Theremin", final_img)

            # input to terminate program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # terminate program
        cap.release()
        cv2.destroyAllWindows()
        stream.stop()
        stream.close()

if __name__ == "__main__":
    main()