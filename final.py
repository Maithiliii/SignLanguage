import subprocess, cv2, math, os, pickle, pygame, pyttsx3, tkinter as tk, mediapipe as mp, numpy as np
from PIL import Image, ImageTk
from ctypes import cast, POINTER
from tkinter import ttk
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from cvzone.HandTrackingModule import HandDetector

model_dict = pickle.load(open('WeSpeak\model.p', 'rb')) #Load the pre-trained letter model
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
    def _init_(self, root):
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

    def show_about_page(self):
        about_text = "\"WeSpeak\" is an innovative project that harnesses the power of computer vision and machine learning to create an interactive and intuitive user experience to bridge the communication gap between differently abled people. \n\nDeveloped using Python and various libraries including OpenCV, Mediapipe, and TensorFlow We Speak offers a range of functionalities aimed at enhancing communication and interaction through hand gestures.\n\nDesigned with usability and accessibility in mind, We Speak aims to revolutionize the way we interact with technology and with each other, offering a glimpse into the future of human-computer interaction.\n\nJoin us on this exciting journey as we explore the endless possibilities of gesture-based communication with We Speak!"
        self.show_info_page("About", about_text)

    def show_team_page(self):
        team_text = "Meet our amazing WeSpeak team members! \n\nLeander Fernandes: \nPrimary back-end developer and Tech Lead. He made critical architectural decisions, and offered insightful suggestions that shaped the direction of the project. His leadership and technical prowess were instrumental in driving the project forward.\n\nRoshan Rizvi: \nPrimary front-end developer. His focus on creating an intuitive frontend interface and his contributions to backend tasks were essential to the project's success. \n\nMaithily Naik: \nMaithily provided valuable contributions to the We Speak project, offering support in coding tasks, training the models during the development stages, and assisting in various aspects throughout the development of the project are irreplacable!"
        self.show_info_page("Team", team_text)

    def show_future_page(self):
        future_text = "*Exciting* things are coming in the future! \n\n-Multi-Language Support: In our efforts to make We Speak accessible to users worldwide, we're exploring the integration of multi-language support. By training models on diverse linguistic datasets, we aspire to enable users to communicate effectively in their preferred language. \n\n-Enhanced Gesture Recognition: We're continuously refining our gesture recognition algorithms to improve accuracy and expand the range of recognizable gestures. By leveraging advanced machine learning techniques and incorporating more comprehensive datasets, we aim to make We Speak even more intuitive and responsive to users' gestures.   \n\n-Refined UI: A more intuitive and easily navigatable UI to make it easier and simpler for new users is currently being developed! "
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
        # Display loading message
        loading_label = tk.Label(self.root, text="Classifying letters... Please wait.", font=("Arial", 12), fg="gray")
        loading_label.pack(pady=10)
        word = ""
        test = ''
        fnum = 0
        caplttr = 0

        try:
            cap = cv2.VideoCapture(0)

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
                        print(word)
                        word = ""

                    if key == ord('t'):
                        # print(f'Current word is  {autocorrect(word)}')
                        # speech.say(autocorrect(word))
                        cv2.putText(image, word, (3, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        speech.runAndWait()
                        print(word)
                        speech.say(word)
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
        # Display loading message
        loading_label = tk.Label(self.root, text="Classifying gesture... Please wait.", font=("Arial", 12), fg="gray")
        loading_label.pack(pady=10)

        try:
            actions = np.array(['Hello', 'Goodbye', 'Please'])

            model = Sequential()
            model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
            model.add(LSTM(128, return_sequences=True, activation='relu'))
            model.add(LSTM(64, return_sequences=False, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(actions.shape[0], activation='softmax'))

            model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

            model.load_weights('WeSpeak\\action.h5')

            # 1. New detection variables
            sequence = []
            sentence = []
            say = []
            threshold = 0.8

            cap = cv2.VideoCapture(0)
            # Set mediapipe model
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    if results.left_hand_landmarks or results.right_hand_landmarks:

                        # 2. Prediction logic
                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        sequence = sequence[-30:]

                        if len(sequence) == 30:
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                            print(actions[np.argmax(res)])

                            # 3. Viz logic
                            if res[np.argmax(res)] > threshold:
                                if len(sentence) > 0:
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    sentence.append(actions[np.argmax(res)])

                            if len(sentence) > 5:
                                sentence = sentence[-5:]

                    else:
                        sequence.clear()

                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.imshow('OpenCV Feed', image) # Show to screen

                    if cv2.waitKey(10) & 0xFF == 27: break # Break gracefully

                cap.release()
                cv2.destroyAllWindows()
                self.show_feedback("Action recognition was completed.")
        except Exception as e: self.show_feedback(f"Error: {str(e)}")
        loading_label.destroy()  # Remove loading message
   
    def run_volume_rocker(self):

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
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
        try:
            # Parameters
            width, height = 1090, 1080
            gestureThreshold = 300
            folderPath = "WeSpeak\presentation"

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
                imgCurrent = cv2.resize(imgCurrent, (1920, 1000))

                cv2.imshow("Slides", imgCurrent)

                key = cv2.waitKey(1)
                if key == 27: break
            cap.release()
            cv2.destroyAllWindows()
        except FileNotFoundError:
            self.show_feedback("Presentation file does not exist.")

        self.show_feedback("Presentation executed successfully.")

    def snake_game(self):
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
            def message(msg, color, score=None):
                mesg = font_style.render(msg, True, color)
                dis.blit(mesg, [dis_width / 6, dis_height / 3])
                if score is not None:
                    score_text = font_style.render("Score: " + str(score), True, color)
                    dis.blit(score_text, [dis_width / 6, dis_height / 3 + 50])

            # Function to display the score
            def show_score(score):
                score_text = font_style.render("Score: " + str(score), True, black)
                dis.blit(score_text, [0, 0])

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
                                quit()
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

                    # for x in snake_List[:-1]:
                    #     if x == snake_Head:
                    #         game_over = True

                    our_snake(snake_block, snake_List)
                    show_score(score)

                    pygame.display.update()

                    if x1 == foodx and y1 == foody:
                        foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
                        foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
                        Length_of_snake += 2
                        score += 10

                    # Game over conditions
                    if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
                        game_over = True

                    clock.tick(snake_speed)

                while game_over:

                    dis.fill(blue)
                    message("You Lost! Press Q-Quit or R-Play Again", red, score)
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

        except Exception as e:
            self.show_feedback(f"Error: {str(e)}")

if _name_ == "_main_":
    root = tk.Tk()
    app = WeSpeakApp(root)
    root.mainloop()
