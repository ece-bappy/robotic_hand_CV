import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Euclidean distance
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# Open the video file
video = cv2.VideoCapture('test.mp4')

while video.isOpened():
    success, image = video.read()
    if not success:
        print("End of video stream.")
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Draw hand landmarks and calculate distances
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Calculate and display distances
            wrist = hand_landmarks.landmark[0]
            for tip_id in [4, 8, 12, 16, 20]:
                tip = hand_landmarks.landmark[tip_id]
                distance = calculate_distance(wrist, tip)
                
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = image.shape
                cx, cy = int(tip.x * w), int(tip.y * h)
                
                # Display distance near the tip of each finger
                cv2.putText(image, f"{distance:.2f}", (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the processed frame
    cv2.imshow('Hand Tracking with Distances', image)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
