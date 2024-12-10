import tkinter as tk
from tkinter import ttk
from tkinter import Canvas
import cv2
import threading
import mediapipe as mp
import pickle
import torch
import numpy as np
from PIL import Image, ImageTk
from mlp import NeuralNetwork
from generate_landmark_data import HandLandmarksDetector


class HandGestureInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Interface")

        # Load the trained model
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.model.eval()

        # Initialize Mediapipe
        self.mp_hands = HandLandmarksDetector(min_detection_confidence=0.5)

        # Initialize Video Capture
        self.cap = None
        self.running = False

        # Set up GUI components
        self.setup_gui()

    def setup_gui(self):
        # Button to open camera
        self.btn_open = ttk.Button(
            self.root, text="Open Camera", command=self.start_camera)
        self.btn_open.pack(pady=10)

        # Canvas to display webcam
        self.canvas = Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        # Canvas for circles
        self.circle_canvas = Canvas(self.root, width=200, height=100)
        self.circle_canvas.pack(pady=10)
        self.circles = [
            self.circle_canvas.create_oval(20, 20, 60, 60, fill="black"),
            self.circle_canvas.create_oval(80, 20, 120, 60, fill="black"),
            self.circle_canvas.create_oval(140, 20, 180, 60, fill="black")
        ]

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.btn_open.config(text="Close Camera", command=self.stop_camera)
            threading.Thread(target=self.process_video, daemon=True).start()
        else:
            self.stop_camera()

    def stop_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            self.btn_open.config(text="Open Camera", command=self.start_camera)

    def process_video(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            annotated_image = frame.copy()

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_landmarks, annotated_image = self.mp_hands.detectHand(
                rgb_frame)

            # Prepare input for the model
            if hands_landmarks:
                input_data = torch.tensor(
                    hands_landmarks, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output = torch.softmax(self.model(input_data), dim=1)

                    # If confidence is low, reset the circles
                    if output.max().item() < 0.5:
                        self.update_circles(0)
                    else:
                        # Get the predicted label
                        _, predicted = torch.max(output, dim=1)
                        prediction = predicted.item()

                        # Update the circles based on the prediction
                        self.update_circles(prediction)
            else:
                self.reset_circles()

            # Convert the image to PhotoImage
            img = Image.fromarray(annotated_image)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

        self.reset_circles()
        self.cap.release()

    def update_circles(self, prediction):
        # Reset all circles to black
        for circle in self.circles:
            self.circle_canvas.itemconfig(circle, fill="black")

        if prediction == 0:
            self.reset_circles()
        elif prediction == 1:
            self.circle_canvas.itemconfig(self.circles[0], fill="yellow")
        elif prediction == 2:
            self.circle_canvas.itemconfig(self.circles[1], fill="yellow")
        elif prediction == 3:
            self.circle_canvas.itemconfig(self.circles[2], fill="yellow")
        elif prediction == 4:
            for circle in self.circles:
                self.circle_canvas.itemconfig(circle, fill="yellow")

    def reset_circles(self):
        for circle in self.circles:
            self.circle_canvas.itemconfig(circle, fill="black")

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureInterface(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
