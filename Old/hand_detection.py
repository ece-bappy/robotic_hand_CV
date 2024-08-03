import cv2
import serial

# Initialize the Arduino connection
ser = serial.Serial("COM5", 9600)  # Replace 'COM3' with your Arduino's serial port

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Perform object detection

    # If the object is detected:
    # Send a signal to the Arduino
    if object_detected:
        ser.write(b"1")  # Send '1' to Arduino to set digital pin 13 high

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()  # Close the Arduino connection when done
