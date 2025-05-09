import numpy as np
import os
import pickle
import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def extract_keypoints(results):
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
    return np.concatenate([lh, rh])


import numpy as np

def extract(results):
    data = []

    # Extract x coordinates for each landmark in the left hand
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            data.append(landmark.x)
    else:
        data.extend([0.0] * 21)  # Append 21 zeros if no left hand landmarks are detected

    # Extract x coordinates for each landmark in the right hand
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            data.append(landmark.x)
    else:
        data.extend([0.0] * 21)  # Append 21 zeros if no right hand landmarks are detected

    # Extract y coordinates for each landmark in the left hand
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            data.append(landmark.y)
    else:
        data.extend([0.0] * 21)  # Append 21 zeros if no left hand landmarks are detected

    # Extract y coordinates for each landmark in the right hand
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            data.append(landmark.y)
    else:
        data.extend([0.0] * 21)  # Append 21 zeros if no right hand landmarks are detected

    return data



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


DATA_DIR = './datatwo'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    # Check if the current item is a directory
    if os.path.isdir(os.path.join(DATA_DIR, dir_)):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
                data_aux = []
                x_ = []
                y_ = []

                img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                image, results = mediapipe_detection(img, holistic)

                # print(extract_keypoints(results))
                # print(np.shape(extract(results)))
                data.append(extract(results))
                labels.append(dir_)



# print(data)
# print(np.shape(data))
# Rest of the code remains the same
f = open('datatwo.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
print("Data arrays were created")