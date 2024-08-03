import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for the palm side (green)
    lower_palm = np.array([0, 20, 40], dtype=np.uint8)
    upper_palm = np.array([25, 255, 255], dtype=np.uint8)

    # Define the range for the back side (red)
    lower_back = np.array([25, 255, 255], dtype=np.uint8)
    upper_back = np.array([25, 255, 255], dtype=np.uint8)

    mask_palm = cv2.inRange(hsv, lower_palm, upper_palm)
    mask_back = cv2.inRange(hsv, lower_back, upper_back)

    # Combine the two masks
    combined_mask = cv2.add(mask_palm, mask_back)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # Find contours in the binary image
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 20000:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if (
                aspect_ratio < 1
            ):  # If the aspect ratio is less than 1, it's likely the back side
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 0, 255), 2
                )  # Red for back
            else:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )  # Green for palm

    # Display the frame
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
