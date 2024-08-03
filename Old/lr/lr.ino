#include <Servo.h>
Servo servo;  // Create a Servo object

int servoPosition = 0;
int ledPin = 13;

void setup() {
  servo.attach(9); 
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);  // Initialize serial communication
}

void loop() {
  if (Serial.available() > 0) {
    char data = Serial.read();
    if (data == '1') {
      digitalWrite(ledPin, HIGH);  // Set the LED on
      servoPosition = 0;
    } else if (data == '0') {
      digitalWrite(ledPin, LOW);   // Set the LED off
      servoPosition = 90;
    }
    servo.write(servoPosition);
  }
}
