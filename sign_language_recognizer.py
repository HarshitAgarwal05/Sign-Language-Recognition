import cv2
import numpy as np
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
import threading

SIGNS = ['A', 'B', 'C']  # You can expand this list
SAMPLES_PER_SIGN = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

class SignLanguageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.current_sign = ''
        self.collecting = False
        self.collect_label = ''
        self.samples = {sign: [] for sign in SIGNS}
        self.labels = []
        self.knn = None
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        self.label = Label(window, text="Recognized Sign: ", font=("Arial", 24))
        self.label.pack()

        self.canvas = Label(window)
        self.canvas.pack()

        self.speak_button = Button(window, text="Speak", command=self.speak_sign, font=("Arial", 16))
        self.speak_button.pack()

        self.info_label = Label(window, text="Press 'a', 'b', 'c' to record samples. Press 't' to train. Press 'q' to quit.", font=("Arial", 12))
        self.info_label.pack()

        self.window.bind('<Key>', self.key_handler)
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def key_handler(self, event):
        key = event.char.lower()
        if key in [s.lower() for s in SIGNS]:
            self.collecting = True
            self.collect_label = key.upper()
        elif key == 't':
            self.train_classifier()
        elif key == 'q':
            self.on_closing()

    def train_classifier(self):
        X = []
        y = []
        for sign, samples in self.samples.items():
            X.extend(samples)
            y.extend([sign] * len(samples))
        if X:
            self.knn = KNeighborsClassifier(n_neighbors=3)
            self.knn.fit(X, y)
            self.info_label.config(text="Classifier trained! Show a sign.")
        else:
            self.info_label.config(text="No samples to train on.")

    def extract_landmarks(self, hand_landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

    def update(self):
        ret, frame = self.vid.read()
        sign = ''
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = self.extract_landmarks(hand_landmarks)
                    if self.collecting and len(self.samples[self.collect_label]) < SAMPLES_PER_SIGN:
                        self.samples[self.collect_label].append(landmarks)
                        self.info_label.config(text=f"Collecting {self.collect_label}: {len(self.samples[self.collect_label])}/{SAMPLES_PER_SIGN}")
                        if len(self.samples[self.collect_label]) >= SAMPLES_PER_SIGN:
                            self.collecting = False
                            self.info_label.config(text=f"Done collecting {self.collect_label}. Press 't' to train.")
                    if self.knn is not None and not self.collecting:
                        sign = self.knn.predict([landmarks])[0]
                        self.current_sign = sign
                        self.label.config(text=f"Recognized Sign: {sign}")
            else:
                if not self.collecting:
                    self.label.config(text="Recognized Sign: ")
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)
        self.window.after(30, self.update)

    def speak_sign(self):
        if self.current_sign:
            threading.Thread(target=speak, args=(self.current_sign,)).start()

    def on_closing(self):
        self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = Tk()
    app = SignLanguageApp(root, "Sign Language Recognition (MediaPipe)") 