import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import sys

class HandTrackingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Tracking GUI")
        self.setGeometry(100, 100, 1000, 700)

        # Main widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Video display
        self.video_label = QLabel()
        main_layout.addWidget(self.video_label)

        # Right side layout for table and calibration status
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)

        # Table for distances and angles
        self.table = QTableWidget(5, 3)
        self.table.setHorizontalHeaderLabels(["Finger", "Distance", "Angle"])
        self.table.verticalHeader().setVisible(False)
        self.table.setItem(0, 0, QTableWidgetItem("Thumb"))
        self.table.setItem(1, 0, QTableWidgetItem("Index"))
        self.table.setItem(2, 0, QTableWidgetItem("Middle"))
        self.table.setItem(3, 0, QTableWidgetItem("Ring"))
        self.table.setItem(4, 0, QTableWidgetItem("Pinky"))
        right_layout.addWidget(self.table)

        # Calibration status
        self.calibration_label = QLabel("Calibrating...")
        right_layout.addWidget(self.calibration_label)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Open the video file
        self.video = cv2.VideoCapture('test1.mp4')

        # Variables for calibration
        self.min_distances = [float('inf')] * 5
        self.max_distances = [0] * 5
        self.calibration_frames = 30
        self.frame_count = 0

        # Start video processing
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)  # 30 ms between frames

    def process_frame(self):
        success, image = self.video.read()
        if not success:
            self.timer.stop()
            return

        # Resize image to 600x600
        image = self.resize_with_padding(image, (600, 600))

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = self.hands.process(image_rgb)

        finger_angles = [90] * 5  # Default to 90 degrees (middle position)
        distances = [0] * 5

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[0]
                for i, tip_id in enumerate([4, 8, 12, 16, 20]):
                    tip = hand_landmarks.landmark[tip_id]
                    distance = self.calculate_distance(wrist, tip)
                    distances[i] = distance

                # Calibration
                if self.frame_count < self.calibration_frames:
                    for i, distance in enumerate(distances):
                        self.min_distances[i] = min(self.min_distances[i], distance)
                        self.max_distances[i] = max(self.max_distances[i], distance)
                    self.frame_count += 1
                    self.calibration_label.setText(f"Calibrating: {self.frame_count}/{self.calibration_frames}")
                else:
                    # Map distances to servo angles
                    finger_angles = [self.map_distance_to_angle(d, min_d, max_d, 0, 180) 
                                     for d, min_d, max_d in zip(distances, self.min_distances, self.max_distances)]
                    self.calibration_label.setText("Calibrated")

                # Update table
                for i in range(5):
                    self.table.setItem(i, 1, QTableWidgetItem(f"{distances[i]:.2f}"))
                    self.table.setItem(i, 2, QTableWidgetItem(f"{finger_angles[i]:.0f}"))

        # Convert image to Qt format and display
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def calculate_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def map_distance_to_angle(self, distance, min_dist, max_dist, min_angle, max_angle):
        return np.interp(distance, [min_dist, max_dist], [max_angle, min_angle])

    def resize_with_padding(self, image, expected_size):
        ih, iw = image.shape[:2]
        eh, ew = expected_size
        scale = min(eh / ih, ew / iw)
        nh = int(ih * scale)
        nw = int(iw * scale)
        image_resized = cv2.resize(image, (nw, nh))
        new_image = np.full((eh, ew, 3), (0, 0, 0), dtype=np.uint8)
        dy = (eh - nh) // 2
        dx = (ew - nw) // 2
        new_image[dy:dy+nh, dx:dx+nw, :] = image_resized
        return new_image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = HandTrackingGUI()
    gui.show()
    sys.exit(app.exec_())
