#include <Servo.h>

Servo servo;  // Create a Servo object

int servoPosition = 0;  // Initial servo position (90 degrees)

void setup() {
  servo.attach(9);  // Attach the servo to digital pin 9
  Serial.begin(9600);  // Initialize serial communication
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == '0') {
      servoPosition = 0;  // Set servo to 0 degrees
    } else if (command == '1') {
      servoPosition = 90;  // Set servo to 90 degrees
    }
    servo.write(servoPosition);  // Send the new position to the servo
       
  }
  delay(500);
}
