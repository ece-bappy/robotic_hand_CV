import cv2
import numpy as np
import serial

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the Arduino connection
ser = serial.Serial("COM5", 9600)  # Replace 'COM5' with your Arduino's serial port

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for the palm side (green)
    lower_palm = np.array([0, 20, 40], dtype=np.uint8)
    upper_palm = np.array([25, 255, 255], dtype=np.uint8)

    mask_palm = cv2.inRange(hsv, lower_palm, upper_palm)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(mask_palm, (5, 5), 0)

    # Find contours in the binary image
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hand_position = None  # Initialize hand position variable

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 20000:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)

            # Determine the position of the hand
            frame_width = frame.shape[1]
            if x + w < frame_width // 2:
                hand_position = "left"
                ser.write(b"1")  # Send '1' to Arduino to set digital pin 13 high
            else:
                hand_position = "right"
                ser.write(b"0")  # Send '0' to Arduino to set digital pin 13 low

                # Control the servo based on hand position
                if hand_position == "left":
                    ser.write(b"0")  # Send '0' to Arduino to set servo to 0 degrees
                else:
                    ser.write(b"1")  # Send '1' to Arduino to set servo to 90 degrees

            # Draw a rectangle around the detected hand
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if hand_position:
        cv2.putText(
            frame, hand_position, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # Display the frame
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()  # Close the Arduino connection when done
