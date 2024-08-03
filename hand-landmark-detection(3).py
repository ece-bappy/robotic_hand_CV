import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Euclidean distance
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# Function to map distance to servo angle
def map_distance_to_angle(distance, min_dist, max_dist, min_angle, max_angle):
    return np.interp(distance, [min_dist, max_dist], [max_angle, min_angle])

# Function to resize image to 600x600 while maintaining aspect ratio
def resize_with_padding(image, expected_size):
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

# Open the video file
video = cv2.VideoCapture('test1.mp4')

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object to save the processed video
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


# Variables to store min and max distances for calibration
min_distances = [float('inf')] * 5
max_distances = [0] * 5
calibration_frames = 60
frame_count = 0

print("Calibrating... Please ensure the video shows various hand positions.")

while video.isOpened():
    success, image = video.read()
    if not success:
        print("End of video.")
        break

    # Resize image to 600x600
    image = resize_with_padding(image, (600, 600))

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Initialize finger_angles
    finger_angles = [90] * 5  # Default to 90 degrees (middle position)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[0]
            distances = []
            for tip_id in [4, 8, 12, 16, 20]:
                tip = hand_landmarks.landmark[tip_id]
                distance = calculate_distance(wrist, tip)
                distances.append(distance)

            # Calibration
            if frame_count < calibration_frames:
                for i, distance in enumerate(distances):
                    min_distances[i] = min(min_distances[i], distance)
                    max_distances[i] = max(max_distances[i], distance)
                frame_count += 1
            else:
                # Map distances to servo angles
                finger_angles = [map_distance_to_angle(d, min_d, max_d, 0, 180) 
                                 for d, min_d, max_d in zip(distances, min_distances, max_distances)]

            # Display distances and angles on the image
            for i, (distance, angle) in enumerate(zip(distances, finger_angles)):
                cv2.putText(image, f"F{i+1}: {distance:.2f} / {angle:.0f}", (10, 30 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display calibration progress
    if frame_count < calibration_frames:
        cv2.putText(image, f"Calibrating: {frame_count}/{calibration_frames}", (10, 570), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(image, "Calibrated", (10, 570), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Hand Tracking', image)

    # Control playback speed
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.waitKey(-1)  # Wait until any key is pressed

# Release resources
video.release()
cv2.destroyAllWindows()
