import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# solution APIs
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

CONVEX_HULL_THRESHOLD = 1100

# Volume Control Library Usage
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, volBar, volPer = volRange[0], volRange[1], 400, 0

# Webcam Setup
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Mediapipe Hand Landmark Model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cam.isOpened():
        success, image = cam.read()
        image=cv2.flip(image,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if results.multi_hand_landmarks:

            num_hands = len(results.multi_hand_landmarks)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_info in results.multi_handedness:
                # Determine handedness
                hand_label = hand_info.classification[0].label


        # multi_hand_landmarks method for Finding postion of Hand landmarks
        lmList = []
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) ==2:
                for i, hand_info in enumerate(results.multi_handedness):

                    # if hand_label == 'Left':
                        # Get landmarks of thumb, index, middle, ring, and pinky fingertips
                        fingertips = [
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                        ]

                        # Extract x, y coordinates of fingertips
                        fingertips_xy = [(int(f.x *image.shape[1]), int(f.y * image.shape[0])) for f in fingertips]

                        # Calculate convex hull
                        hull = cv2.convexHull(np.array(fingertips_xy), returnPoints=True)

                        # Calculate the area of the convex hull
                        hull_area = cv2.contourArea(hull)

                        # Check if the area is below the threshold to identify a fist
                        if hull_area < CONVEX_HULL_THRESHOLD:
                        # if hand_info.classification[0].label == 'Right' and hull_area < CONVEX_HULL_THRESHOLD:
                            myHand = results.multi_hand_landmarks[i]
                            for id, lm in enumerate(myHand.landmark):
                                h, w, c = image.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                lmList.append([id, cx, cy])

                                # Assigning variables for Thumb and Index finger position
                            if len(lmList) != 0:
                                x1, y1 = lmList[4][1], lmList[4][2]
                                x2, y2 = lmList[8][1], lmList[8][2]

                                # Marking Thumb and Index finger
                                cv2.circle(image, (x1, y1), 15, (255, 255, 255))
                                cv2.circle(image, (x2, y2), 15, (255, 255, 255))
                                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                length = math.hypot(x2 - x1, y2 - y1)
                                if length < 3:
                                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

                                vol = np.interp(length, [3, 120], [minVol, maxVol])
                                volume.SetMasterVolumeLevel(vol, None)
                                volBar = np.interp(length, [3, 120], [400, 150])
                                volPer = np.interp(length, [3, 120], [0, 100])



            # Volume Bar
            # cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
            # cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
            # cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
            #             1, (0, 0, 0), 3)

        cv2.imshow('handDetector', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cam.release()