
#    this is the most stable and latest version with everything working nicely
#    this version has all the methods load on startup so its quicker while running 

import cv2, math, os, pickle, pygame, pyttsx3, tkinter as tk, mediapipe as mp, numpy as np
from ctypes import cast, POINTER
from tkinter import ttk
import random
import time
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from cvzone.HandTrackingModule import HandDetector

# Load the pre-trained model
model_dict = pickle.load(open('./weights/model.p', 'rb'))
model = model_dict['model']


mp_holistic = mp.solutions.holistic # MediaPipe Holistic model initialization
mp_drawing = mp.solutions.drawing_utils
speech = pyttsx3.init()# Labels mapping

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

def is_left_hand_fist(results, image): # VolumeCOntrols
    if results.left_hand_landmarks:
        # Get landmarks of thumb, index, middle, ring, and pinky fingertips
        fingertips = [
            results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP],
            results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP],
            results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP],
            results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP],
            results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP],
        ]

        # Extract x, y coordinates of fingertips
        fingertips_xy = [(int(f.x * image.shape[1]), int(f.y * image.shape[0])) for f in fingertips]

        # Calculate convex hull
        hull = cv2.convexHull(np.array(fingertips_xy), returnPoints=True)

        # Calculate the area of the convex hull
        hull_area = cv2.contourArea(hull)

        # Check if the area is below the threshold to identify a fist
        if hull_area < 1100:
            return True
    return False

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                            )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )

def extract(results): #Letter reg
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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


class WeSpeakApp:
    def __init__(self, root):
        self.root = root
        self.root.title("We Speak")

        # Main title label with font and color changes
        title_label = tk.Label(root, text="We Speak", font=("Arial", 60), fg="navy")
        title_label.pack(pady=20, padx=(180, 0))
        
        button_frame = tk.Frame(root) # Frame to contain the main buttons
        button_frame.pack()
        
        new_buttons_frame = tk.Frame(root)  # Frame to contain the side buttons
        new_buttons_frame.pack(side=tk.LEFT, anchor=tk.NW)  # Anchor to the top-left corner

        # Themed buttons with color changes
        style = ttk.Style()
        style.configure("TButton", padding=10, font=("Arial", 12))

        letters_button = ttk.Button(button_frame, text="Letters", compound=tk.TOP, command=self.run_letters, style="TButton")
        letters_button.pack(side=tk.LEFT, padx=(180, 20))

        gesture_button = ttk.Button(button_frame, text="Gestures", compound=tk.TOP, command=self.run_gesture, style="TButton")
        gesture_button.pack(side=tk.LEFT, padx=(0, 20))
        
        # Add side buttons
        self.add_new_button(new_buttons_frame, "Learn ISL", self.run_learn_isl)
        self.add_new_button(new_buttons_frame, "Volume Control", self.run_volume_rocker)
        self.add_new_button(new_buttons_frame, "Presentation", self.run_presentation)
        self.add_new_button(new_buttons_frame, "Snake Game", self.snake_game)
        self.add_new_button(new_buttons_frame, "About", self.show_about_page)
        self.add_new_button(new_buttons_frame, "Team", self.show_team_page)
        self.add_new_button(new_buttons_frame, "Future", self.show_future_page)
        
        root.geometry("800x600") # Set initial window size
        root.resizable(False, False) # Disable resizing
        self.current_info_frame = None  # Instance variable to keep track of the current info frame
        
    def add_new_button(self, frame, text, command):
        new_button = ttk.Button(frame, text=text, compound=tk.TOP, command=command, style="TButton")
        new_button.pack(side=tk.TOP, padx=10, pady=5, anchor=tk.W)  # Anchor to the left

    def show_letters_page(self):
        letters_text = "Step-by-step tutorial on how to use WeSpeak for letters:\n\n1. Ensure that your device's camera is facing towards you and positioned to capture your hand gestures clearly. Find a well-lit environment with minimal distractions for optimal detection accuracy.\n2. Begin signing individual letters from the alphabet using the appropriate hand shapes and movements. Hold each letter gesture steady within the camera's view to allow the app to accurately recognize and interpret the sign.\n3. As you sign each letter, the app will display the corresponding letter on the screen in real-time.\n4. Once a letter is successfully detected and displayed on the screen, the app will convert it into audible speech."
        # letters_text = "This is one of the main functionalities of the app, it helps users spell out words in Indian Sign Language (ISL) with real-time interpretion of gestures using computer vision, thus bridiging the gap between humans and technology.\n\nUpon activation, users are seamlessly guided to a dedicated page where they can engage in letter recognition activities. \n\nTo help users spell words quicker and easier, we have implemented an autocorrect feature that aids you and enables you to communicate more efficiently.\n\nThere is also a text-to-speech function that helps ease into the computer-human interaction to facilitate better communication!"
        self.show_info_page("Indian Sign Language", letters_text)

    def show_gesture_page(self):
        gesture_text = "Step-by-step tutorial on how to use WeSpeak for actions/gestures:\n\n1. Ensure that your device's camera is facing towards you and positioned to capture your hand gestures clearly. Find a well-lit environment with minimal distractions for optimal detection accuracy.\n2. Begin performing sign language actions or gestures. Use appropriate hand shapes, movements, and facial expressions to convey meaning effectively.\n3. As you perform each action, the app will analyse and interpret your gestures in real-time. The detected actions will be displayed on the screen as they are recognized by the app.\n4. Once an action is successfully detected and displayed on the screen, the app will convert it into audible speech."

        # gesture_text = "This is one of the main functionalities of this app, it helps users to act out the words in sign language and it will translate it into plane text using computer vision, machine learning, and classification algorithms.\n\nThis function opens up a dedicated window where the user can see themselves and the predictions for the actions performed. Beyond mere recognition, our application also emphasizes user empowerment and engagement. By offering instant feedback and visual confirmation of recognized signs, users can communicate confidently, knowing that their expressions are accurately interpreted. Moreover, the application fosters inclusivity by providing learning resources and feedback mechanisms, enabling users to improve their sign language skills progressively."
        self.show_info_page("Gesture Communication", gesture_text)

    def show_learn_ISL_page(self):
        gesture_text = "A step by step application for learning the letters in the Indian Sign Language with demonstration pictures. \nHappy learning!"
        # gesture_text = "This is one of the main functionalities of this app, it helps users to act out the words in sign language and it will translate it into plane text using computer vision, machine learning, and classification algorithms.\n\nThis function opens up a dedicated window where the user can see themselves and the predictions for the actions performed. Beyond mere recognition, our application also emphasizes user empowerment and engagement. By offering instant feedback and visual confirmation of recognized signs, users can communicate confidently, knowing that their expressions are accurately interpreted. Moreover, the application fosters inclusivity by providing learning resources and feedback mechanisms, enabling users to improve their sign language skills progressively."
        self.show_info_page("Gesture Communication", gesture_text)

    def show_volume_page(self):
        volume_text = "Step-by-step tutorial on how to use WeSpeak for hand gesture volume control:\n\n1. Ensure that your device's camera is facing towards you and positioned to capture your hand gestures clearly. Find a well-lit environment with minimal distractions for optimal detection accuracy.\n2. Make a fist with one hand, while using the index finger and thumb of the other hand to control volume. Adjust the volume by moving the index finger and thumb closer together to decrease volume, or further apart to increase volume. The distance between the thumb and index finger indicates the volume level.\n3. As you adjust the distance between your index finger and thumb, the app will analyze and interpret your hand movements in real-time. The volume level will adjust accordingly, either increasing or decreasing based on the distance between your fingers.\n4. Monitor the device's volume level as you adjust the distance between your fingers."
        self.show_info_page("Volume Controller", volume_text)

    def show_presentation_page(self):
        presentation_text = "Step-by-step tutorial for using a hand gesture-controlled presentation:\n\n1. Ensure that your devices camera is facing towards you and positioned to capture your hand gestures clearly. Find a well-lit environment with minimal distractions for optimal detection accuracy.\n2. Use one finger to control the pointer on the slides. Point your finger towards the screen to move the pointer in the desired direction.\n3. Use two fingers to activate the drawing mode. You can then draw or annotate directly on the slides by moving your fingers across the screen\n4. Navigate through slides by using the following gestures:\n\tPrevious slide: Stick thumb out\n\tNext Slide: Stick pinky out \n\tLazer pointer: Point with your index finger \n\tDraw: Use your index and middle finger to draw \n\tErase drawing: Hold up your index, middle, and ring finger together."
        # presentation_text = "Allows for effortless, comfortable and handfree presentation control using hand gestures. \n\nControls: \n\tPrevious slide: Stick thumb out\n\tNext Slide: Stick pinky out \n\tLazer pointer: Point with your index finger \n\tDraw: Use your index and middle finger to draw \n\tErase drawing: Hold up your index, middle, and ring finger together"
        self.show_info_page("Presentation Controller", presentation_text)

    def show_snake_page(self):
        snake_text = "This is an addictive and fun game using computer vision and hand recognition using image processing to control the snake as it tried to collect all of it food to grow. \n\nBased on the classic game we all played when we were younger, but now modernized using hand controls to bridge the past and future together in a new, fun and interesting way! \n\n CONTROLS: \n\tmove LEFT: stick thumb out. \n\tmove RIGHT: stick pinky finger out \n\tmove UP: index finger pointing up \n\tmove DOWN: close fist\n\n\n\nHSSSSSSSSSS"
        self.show_info_page("Snake Game", snake_text)

    def show_about_page(self):
        about_text = "\"WeSpeak\" is an innovative project that harnesses the power of computer vision and machine learning to create an interactive and intuitive user experience to bridge the communication gap between differently abled people. \n\nDeveloped using Python and various libraries including OpenCV, Mediapipe, and TensorFlow We Speak offers a range of functionalities aimed at enhancing communication and interaction through hand gestures.\n\nDesigned with usability and accessibility in mind, We Speak aims to revolutionize the way we interact with technology and with each other, offering a glimpse into the future of human-computer interaction.\n\nJoin us on this exciting journey as we explore the endless possibilities of gesture-based communication with We Speak!"
        self.show_info_page("About", about_text)

    def show_team_page(self):
        team_text = "Meet our amazing WeSpeak team members! \n\nLeander Fernandes: \nPrimary back-end developer and Tech Lead. He made critical architectural decisions, and offered insightful suggestions that shaped the direction of the project. His leadership and technical prowess were instrumental in driving the project forward.\n\nRoshan Rizvi: \nPrimary front-end developer. His focus on creating an intuitive frontend interface and his contributions to backend tasks were essential to the project's success. \n\nMaithily Naik: \nMaithily provided valuable contributions to the We Speak project, offering support in coding tasks, training the models during the development stages, and assisting in various aspects throughout the development of the project are irreplacable!"
        self.show_info_page("Team", team_text)

    def show_future_page(self):
        future_text = "**Exciting** things are coming in the future! \n\n-Multi-Language Support: In our efforts to make We Speak accessible to users worldwide, we're exploring the integration of multi-language support. By training models on diverse linguistic datasets, we aspire to enable users to communicate effectively in their preferred language. \n\n-Enhanced Gesture Recognition: We're continuously refining our gesture recognition algorithms to improve accuracy and expand the range of recognizable gestures. By leveraging advanced machine learning techniques and incorporating more comprehensive datasets, we aim to make We Speak even more intuitive and responsive to users' gestures.   \n\n-Refined UI: A more intuitive and easily navigatable UI to make it easier and simpler for new users is currently being developed! "
        self.show_info_page("Future", future_text)

    def show_info_page(self, title, content):
        if self.current_info_frame:   # Destroy the current info frame if it exists
            self.current_info_frame.destroy()

        info_frame = tk.Frame(self.root) # Create a new info frame
        info_frame.pack()

        info_label = tk.Label(info_frame, text=title, font=("Sans Serif", 20, "bold"))
        info_label.pack(padx=10, pady=10)

        info_text = tk.Text(info_frame, wrap=tk.WORD,font=("Sans Serif", 11,), width=70, height=15)
        info_text.insert(tk.END, content)
        info_text.config(state=tk.DISABLED)  # Make it read-only
        info_text.pack(padx=40, pady=20)

        self.current_info_frame = info_frame  # Update the current info frame

    def add_hover_effects(self, button, hover_color):
        button.bind('<Enter>', lambda event, button=button: self.on_enter(event, button, hover_color))
        button.bind('<Leave>', lambda event, button=button: self.on_leave(event, button, "#3498db"))

    def on_enter(self, event, button, hover_color):
        button.config(bg=hover_color)

    def on_leave(self, event, button, original_color):
        button.config(bg=original_color)

    def show_feedback(self, message):
        feedback_label = tk.Label(self.root, text=message, font=("Arial", 12), fg="green")
        feedback_label.pack(pady=10)
        # After a few seconds, remove the feedback label
        self.root.after(3000, feedback_label.destroy)

    def run_letters(self):

        self.show_letters_page()

        # Display loading message
        loading_label = tk.Label(self.root, text="Classifying letters... Please wait.", font=("Arial", 12), fg="gray")
        loading_label.pack(pady=10)
        try:

            word = ""
            test = ''
            fnum = 0
            caplttr = 0

            speech = pyttsx3.init()

            # Load the pre-trained model
            model_dict = pickle.load(open('./weights/model.p', 'rb'))
            model = model_dict['model']

            # Video capture
            cap = cv2.VideoCapture(0)

            # MediaPipe Holistic model initialization
            mp_holistic = mp.solutions.holistic
            mp_drawing = mp.solutions.drawing_utils

            # Labels mapping
            labels_dict = {
                0: 'A',
                1: 'B',
                2: 'C',
                3: 'D',
                4: 'E',
                5: 'F',
                6: 'G',
                7: 'H',
                8: 'I',
                9: 'J',
                10: 'K',
                11: 'L',
                12: 'M',
                13: 'N',
                14: 'O',
                15: 'P',
                16: 'Q',
                17: 'R',
                18: 'S',
                19: 'T',
                20: 'U',
                21: 'V',
                22: 'W',
                23: 'X',
                24: 'Y',
                25: 'Z'
            }

            # Add more labels as needed

            # Autocorrect
            def create_bigram(word):
                return [word[i] + word[i + 1] for i in range(len(word) - 1)]

            def get_similarity_ratio(word1, word2):
                word1, word2 = word1.lower(), word2.lower()
                common = []
                bigram1, bigram2 = create_bigram(word1), create_bigram(word2)

                for i in range(len(bigram1)):
                    # common element
                    try:
                        cmn_elt = bigram2.index(bigram1[i])
                        common.append(bigram1[i])
                    except:
                        continue
                return len(common) / max(len(bigram1), len(bigram2))

            def autocorrect(word,
                            database={'APPLE', 'ANIME', 'BOOK', 'BOOKS', 'BEACH', 'BOOTS', 'BEE', 'BREAD', 'BOARD',
                                      'BENCH', 'BIKE', 'CAR', 'CARS', 'CAT', 'CATS', 'CART', 'CAMP',
                                      'CART', 'DEAD', 'DOG', 'DEATH', 'DRUMS', 'DOOM', 'DOOR', 'EEL', 'EMAIL', 'CODE',
                                      'COMPUTERS', 'CHAT', 'FISH', 'FRIEND', 'FUNCTION', 'FUN',
                                      'FISH', 'DUCK', 'LEE', 'MAITHILI', 'ROSH', 'ROSHAN', 'LEANDER', 'RAZIA', 'WATER',
                                      'KEY', 'LAPTOP', 'PCCE', 'MECH', 'IDEA', 'MARVN', 'WATER',
                                      'EXAM', 'TEST', 'REVIEW', 'HELP', 'YUM', 'LUNCH', 'CANTEEN', 'BOY', 'GIRL',
                                      'SIGN'}, sim_thr4eshold=0.40):
                max_sim = 0.0
                most_sim_word = word

                for data_word in database:
                    cur_sim = get_similarity_ratio(word, data_word)
                    if cur_sim > max_sim:
                        max_sim = cur_sim
                        most_sim_word = data_word

                return f'  {most_sim_word if max_sim > sim_thr4eshold else word}'

            def extract_keypoints(results):
                lh = np.array([[res.x, res.y] for res in
                               results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
                    21 * 2)
                rh = np.array([[res.x, res.y] for res in
                               results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
                    21 * 2)
                return np.concatenate([lh, rh])

            def mediapipe_detection(image, model):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
                image.flags.writeable = False  # Image is no longer writeable
                results = model.process(image)  # Make prediction
                image.flags.writeable = True  # Image is now writeable
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
                return image, results

            def draw_styled_landmarks(image, results):
                # Draw left hand connections
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                          )
                # Draw right hand connections
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

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

            # Main loop for real-time prediction
            while True:
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to retrieve frame from the camera.")
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    data_aux = extract(results)
                    # if (data_aux != np.zeros(126, )).all():
                    if results.left_hand_landmarks or results.right_hand_landmarks:
                        # print("Data shape:", np.shape(data_aux))

                        prediction = model.predict([data_aux])
                        predicted_character = labels_dict[int(prediction[0])]

                        if predicted_character == test and predicted_character != ' ':
                            fnum += 1
                            if fnum >= 10:
                                word += predicted_character
                                fnum = 0
                                caplttr = 5

                        if predicted_character != test:
                            test = predicted_character
                            fnum = 0
                            i = 0

                        # print("Predicted character:", predicted_character)

                        # Draw predicted character on the frame
                        # cv2.rectangle(image, (40, 80), (130, 160), (0, 255, 0), 2)
                        if caplttr > 0:
                            cv2.rectangle(image, (40, 80), (130, 160), (0, 255, 0), 4)
                            caplttr -= 1

                        else:
                            cv2.rectangle(image, (40, 80), (130, 160), (0, 0, 0), 4)

                        cv2.putText(image, predicted_character, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
                        # print(cv2.getTextSize(predicted_character, cv2.FONT_HERSHEY_SIMPLEX, 3, 3))
                    cv2.rectangle(image, (0, 0), (640, 40), (255, 255, 255), -1)
                    cv2.putText(image, word, (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                    # Display the frame with predictions
                    cv2.imshow('Hand Gesture Recognition', image)

                    # Check for key press events
                    key = cv2.waitKey(1)
                    speech.runAndWait()
                    if key == 27:  # Press Esc key to exit
                        break

                    if key == ord('p'):
                        # print(f'Current word is  {autocorrect(word)}')
                        print(autocorrect(word))
                        word = ""

                    if key == ord('t'):
                        # print(f'Current word is  {autocorrect(word)}')
                        # speech.say(autocorrect(word))
                        cv2.putText(image, word, (3, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        speech.runAndWait()
                        print(autocorrect(word))
                        speech.say(autocorrect(word))
                        time.sleep(2)
                        word = ""

            # Release resources
            cap.release()
            cv2.destroyAllWindows()

            self.show_feedback("Letter recognition was completed.")

        except Exception as e:
            self.show_feedback(f"Error: {str(e)}")

        # Remove loading message
        loading_label.destroy()

    def run_gesture(self):
        self.show_gesture_page()
        # Display loading message
        loading_label = tk.Label(self.root, text="Classifying gesture... Please wait.", font=("Arial", 12), fg="gray")
        loading_label.pack(pady=10)

        try:
            actions = np.array(['Hello', 'Indian', 'Sign', 'Language', 'Welcome', 'We', 'Speak', 'Detection'])

            model = Sequential()
            model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20, 258)))
            model.add(LSTM(128, return_sequences=True, activation='relu'))
            model.add(LSTM(64, return_sequences=False, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(actions.shape[0], activation='softmax'))

            model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

            model.load_weights('./weights/action.h5')

            # 1. New detection variables
            sequence = []
            sentence = []
            say = []
            threshold = 0.9

            cap = cv2.VideoCapture(0)
            # Set mediapipe model
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    # print(results)
                    draw_styled_landmarks(image, results)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    if results.left_hand_landmarks or results.right_hand_landmarks:

                        # 2. Prediction logic
                        keypoints = extract_keypoints(results)
                        #         sequence.insert(0,keypoints)
                        #         sequence = sequence[:30]
                        sequence.append(keypoints)
                        sequence = sequence[-20:]

                        if len(sequence) == 20:
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                            # print(actions[np.argmax(res)])

                            # 3. Viz logic
                            if res[np.argmax(res)] > threshold:
                                if len(sentence) > 0:
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                        sequence.clear()
                                else:
                                    sentence.append(actions[np.argmax(res)])
                                    sequence.clear()

                            if len(sentence) > 5:
                                sentence = sentence[-5:]

                            # Viz probabilities
                            # image = prob_viz(res, actions, image, colors)

                    else:
                        sequence.clear()
                        # sentence.clear()

                    cv2.rectangle(image, (0, 0), (640, 40), (255, 255, 255), -1)
                    cv2.putText(image, ' '.join(sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                    # Break gracefully
                    key = cv2.waitKey(1)
                    speech.runAndWait()
                    if key == 27:  # Press Esc key to exit
                        break

                    # if key == ord('p'):
                    #     # print(f'Current word is  {autocorrect(word)}')
                    #     print(autocorrect(word))
                    #     word = ""

                    if key == ord('t'):
                        # print(f'Current word is  {autocorrect(word)}')
                        # speech.say(autocorrect(word))
                        # cv2.putText(image, word, (3, 30),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        speech.runAndWait()
                        # print(autocorrect(word))
                        speech.say(sentence)
                        # time.sleep(2)
                        sentence.clear()

                cap.release()
                cv2.destroyAllWindows()
                self.show_feedback("Action recognition was completed.")
        except Exception as e: self.show_feedback(f"Error: {str(e)}")
        loading_label.destroy()  # Remove loading message


    def run_learn_isl(self):
        self.show_learn_ISL_page()
        # Assuming DATA_PATH is the path to your folder containing images
        DATA_PATH = 'Letters'
        image_files = os.listdir(DATA_PATH)
        num_images = len(image_files)
        current_image_index = 0

        # Initialize variables
        green = 0
        i = 0
        fnum = 0

        # Load the pre-trained model
        model_dict = pickle.load(open('./weights/model.p', 'rb'))
        model = model_dict['model']

        # Video capture
        cap = cv2.VideoCapture(0)

        # MediaPipe Holistic model initialization
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils

        # Labels mapping
        labels_dict = {
           0: 'A',
           1: 'B',
           2: 'C',
           3: 'D',
           4: 'E',
           5: 'F',
           6: 'G',
           7: 'H',
           8: 'I',
           9: 'J',
           10: 'K',
           11: 'L',
           12: 'M',
           13: 'N',
           14: 'O',
           15: 'P',
           16: 'Q',
           17: 'R',
           18: 'S',
           19: 'T',
           20: 'U',
           21: 'V',
           22: 'W',
           23: 'X',
           24: 'Y',
           25: 'Z'
        }

        # Main loop for real-time prediction
        while True:
           with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
               ret, frame = cap.read()
               if not ret:
                   print("Failed to retrieve frame from the camera.")
                   break

               image, results = mediapipe_detection(frame, holistic)
               draw_styled_landmarks(image, results)
               data_aux = extract(results)
               test = labels_dict[i]

               # Display the image
               image_path = os.path.join(DATA_PATH, image_files[current_image_index])
               display_image = cv2.imread(image_path)
               display_image = cv2.resize(display_image, (200, 200))  # Resize the image

               # Overlay the resized image onto the live feed at the top right corner
               frame_height, frame_width, _ = image.shape
               top_right_x = frame_width - display_image.shape[1] - 10  # Subtracting the width of the overlay image
               top_right_y = 10  # Adjust this value to position the image vertically

               # Ensure that the area to replace has the same shape as the display image
               image[top_right_y:top_right_y + display_image.shape[0],
               top_right_x:top_right_x + display_image.shape[1]] = display_image

               if green >= 5:
                   cv2.putText(image, 'Correct!!', (20, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 2)
                   cv2.rectangle(image, (40, 80), (130, 160), (0, 255, 0), 2)
                   green -= 1

                   if green == 5:
                       i += 1
               else:
                   cv2.rectangle(image, (40, 80), (130, 160), (0, 0, 0), 2)
                   cv2.putText(image, 'Now...', (20, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)

               cv2.putText(image, test, (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0), 2)
               cv2.putText(image, 'Try!', (460, 250), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 2)

               if results.left_hand_landmarks or results.right_hand_landmarks:
                   prediction = model.predict([data_aux])
                   predicted_character = labels_dict[int(prediction[0])]
                   print(predicted_character)

                   if predicted_character == test:
                       fnum += 1
                       if fnum >= 10:
                           fnum = 0
                           current_image_index += 1
                           green = 10

               # Display the frame with predictions
               cv2.imshow('Hand Gesture Recognition', image)

               # Check for key press events
               key = cv2.waitKey(1)
               if key == 27:  # Press Esc key to exit
                   break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


    def run_volume_rocker(self):
            self.show_volume_page()

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volRange = volume.GetVolumeRange()
            minVol, maxVol, volBar, volPer = volRange[0], volRange[1], 400, 0
            lmList = []

            # Webcam Setup
            wCam, hCam = 640, 480
            cam = cv2.VideoCapture(0)
            cam.set(3, wCam)
            cam.set(4, hCam)

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cam.isOpened():
                    ret, frame = cam.read()
                    if not ret:
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    # Draw hand landmarks
                    draw_styled_landmarks(image, results)

                    if results.left_hand_landmarks:
                        # Check if left hand is in a fist
                        if is_left_hand_fist(results, image):
                            # Process hand landmarks for volume control
                            lmList.clear()
                            if results.right_hand_landmarks:
                                myHand = results.right_hand_landmarks
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
                                    cv2.line(image, (x1, y1), (x2, y2), (235, 206, 135), 3)
                                    length = math.hypot(x2 - x1, y2 - y1)
                                    if length < 3:
                                        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

                                    # Adjust volume based on finger length
                                    vol = np.interp(length, [3, 120], [minVol, maxVol])
                                    volume.SetMasterVolumeLevel(vol, None)

                                    # Adjust volume bar proportionally to the volume
                                    volBar = int(np.interp(vol, [minVol, maxVol], [50, 600]))

                                    # Calculate volume percentage
                                    volPer = np.interp(vol, [minVol, maxVol], [0, 100])
                    # Draw volume bar
                    cv2.rectangle(image, (50, 20), (volBar, 40), (220, 0, 0), cv2.FILLED)  # Volume bar fill
                    cv2.rectangle(image, (50, 20), (600, 40), (0, 0, 0), 3)  # Volume bar outline
                    cv2.putText(image, f'Volume: {int(volPer)} %', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                                3)  # Volume percentage

                    cv2.imshow('Hand Tracking', image)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            cam.release()
            cv2.destroyAllWindows()
            self.show_feedback("Volume Control executed successfully.")

    def run_presentation(self):
        self.show_presentation_page()
        try:

            # Parameters
            width, height = 1535, 900
            gestureThreshold = 300
            folderPath = "Presentation"

            # Camera Setup
            cap = cv2.VideoCapture(0)
            cap.set(3, width)
            cap.set(4, height)

            # Hand Detector
            detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

            # Variables
            imgList = []
            delay = 30
            buttonPressed = False
            counter = 0
            drawMode = False
            imgNumber = 0
            delayCounter = 0
            annotations = [[]]
            annotationNumber = -1
            annotationStart = False
            hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

            # Get list of presentation images
            pathImages = sorted(os.listdir(folderPath), key=len)
            print(pathImages)

            while True:
                # Get image frame
                success, img = cap.read()
                img = cv2.flip(img, 1)
                pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
                imgCurrent = cv2.imread(pathFullImage)

                # Find the hand and its landmarks
                hands, img = detectorHand.findHands(img)  # with draw
                # Draw Gesture Threshold line
                cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 300, 0), 10)

                if hands and buttonPressed is False:  # If hand is detected

                    hand = hands[0]
                    cx, cy = hand["center"]
                    lmList = hand["lmList"]  # List of 21 Landmark points
                    fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

                    # Constrain values for easier drawing
                    xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
                    yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
                    indexFinger = xVal, yVal

                    if cy <= gestureThreshold:  # If hand is at the height of the face
                        if fingers == [1, 0, 0, 0, 0]:
                            print("Left")
                            buttonPressed = True
                            if imgNumber > 0:
                                imgNumber -= 1
                                annotations = [[]]
                                annotationNumber = -1
                                annotationStart = False
                        if fingers == [0, 0, 0, 0, 1]:
                            print("Right")
                            buttonPressed = True
                            if imgNumber < len(pathImages) - 1:
                                imgNumber += 1
                                annotations = [[]]
                                annotationNumber = -1
                                annotationStart = False

                    if fingers == [0, 1, 0, 0, 0]:
                        cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

                    if fingers == [0, 1, 1, 0, 0]:
                        if annotationStart is False:
                            annotationStart = True
                            annotationNumber += 1
                            annotations.append([])
                        print(annotationNumber)
                        annotations[annotationNumber].append(indexFinger)
                        cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

                    else:
                        annotationStart = False

                    if fingers == [0, 1, 1, 1, 0]:
                        if annotations:
                            annotations.pop(-1)
                            annotationNumber -= 1
                            buttonPressed = True

                else:
                    annotationStart = False

                if buttonPressed:
                    counter += 1
                    if counter > delay:
                        counter = 0
                        buttonPressed = False

                for i, annotation in enumerate(annotations):
                    for j in range(len(annotation)):
                        if j != 0:
                            cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

                imgSmall = cv2.resize(img, (ws, hs))
                h, w, _ = imgCurrent.shape
                imgCurrent = cv2.resize(imgCurrent, (1532, 850))

                # imgCurrent[0:hs, w - ws: w] = imgSmall

                cv2.imshow("Slides", imgCurrent)
                # cv2.imshow("Image", img)

                key = cv2.waitKey(1)
                if key == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

        except FileNotFoundError:
            self.show_feedback("Presentation file does not exist.")

        self.show_feedback("Presentation executed successfully.")

    def snake_game(self):
        self.show_snake_page()
        try:
            # CV2
            cap = cv2.VideoCapture(0)
            # Hand Detector
            detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

            pygame.init()

            # Define colors
            white = (255, 255, 255)
            yellow = (255, 255, 102)
            black = (0, 0, 0)
            red = (213, 50, 80)
            green = (0, 255, 0)
            blue = (50, 153, 213)

            # Set display width and height
            dis_width = 800
            dis_height = 600

            dis = pygame.display.set_mode((dis_width, dis_height))
            pygame.display.set_caption('Snake Game')

            clock = pygame.time.Clock()

            snake_block = 10
            snake_speed = 20

            font_style = pygame.font.SysFont(None, 50)

            # Function to display the snake
            def our_snake(snake_block, snake_list):
                for x in snake_list:
                    pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])

            # Function to display the message
            def message(msg, color, position, score=None):
                mesg = font_style.render(msg, True, color)
                text_rect = mesg.get_rect(center=(dis_width // 2, position))
                dis.blit(mesg, text_rect)
                if score is not None:
                    score_text = font_style.render("Your Score: " + str(score), True, color)
                    score_rect = score_text.get_rect(center=(dis_width // 2, position + 50))
                    dis.blit(score_text, score_rect)

            # Function to display the score
            def show_score(score):
                score_text = font_style.render("Your Score: " + str(score), True, black)
                dis.blit(score_text, [0, 0])

            # Function to load the highest score
            def load_high_score():
                if os.path.exists("highscore.txt"):
                    with open("highscore.txt", "r") as file:
                        return int(file.read())
                else:
                    return 0

            # Function to save the highest score
            def save_high_score(score):
                with open("highscore.txt", "w") as file:
                    file.write(str(score))

            # Main function
            def gameLoop():
                game_over = False

                x1 = dis_width / 2
                y1 = dis_height / 2

                x1_change = 0
                y1_change = 0

                snake_List = []
                Length_of_snake = 1

                foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
                foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

                score = 0
                highest_score = load_high_score()

                while not game_over:
                    # Get image frame
                    success, img = cap.read()
                    img = cv2.flip(img, 1)
                    hands, img = detectorHand.findHands(img)
                    if hands:
                        hand = hands[0]
                        fingers = detectorHand.fingersUp(hand)

                        # Control the snake with hand gestures
                        if fingers == [1, 0, 0, 0, 0]:  # Move left
                            x1_change = -snake_block
                            y1_change = 0
                        elif fingers == [0, 0, 0, 0, 1]:  # Move right
                            x1_change = snake_block
                            y1_change = 0
                        elif fingers == [0, 1, 0, 0, 0]:  # Move up
                            y1_change = -snake_block
                            x1_change = 0
                        elif fingers == [0, 0, 0, 0, 0]:  # Move down
                            y1_change = snake_block
                            x1_change = 0

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            game_over = True
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:  # Quit the game
                                pygame.quit()
                            elif event.key == pygame.K_r:  # Restart the game
                                gameLoop()

                    x1 += x1_change
                    y1 += y1_change
                    dis.fill(blue)
                    pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
                    snake_Head = []
                    snake_Head.append(x1)
                    snake_Head.append(y1)
                    snake_List.append(snake_Head)
                    if len(snake_List) > Length_of_snake:
                        del snake_List[0]

                    our_snake(snake_block, snake_List)
                    show_score(score)

                    pygame.display.update()

                    if x1 == foodx and y1 == foody:
                        foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
                        foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
                        Length_of_snake += 2
                        score += 10

                        if score > highest_score:
                            highest_score = score
                            message("New High Score!", red, dis_height // 3 - 50)

                    # Game over conditions
                    if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
                        game_over = True

                    clock.tick(snake_speed)

                save_high_score(highest_score)

                while game_over:

                    dis.fill(blue)
                    message("High Score: " + str(highest_score), green,
                            dis_height // 3 - 50)  # Display high score at the middle
                    message("You Lost! Press Q-Quit or R-Play Again", red, dis_height // 3, score)
                    pygame.display.update()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            game_over = True
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:  # Quit the game
                                pygame.quit()
                            elif event.key == pygame.K_r:  # Restart the game
                                game_over = False
                                gameLoop()

            gameLoop()
            self.show_feedback("Snake game was closed.")

        except Exception as e:
            self.show_feedback(f"Thank you for playing Snakegame.")

if __name__ == "__main__":
    root = tk.Tk()
    app = WeSpeakApp(root)
    root.mainloop()
