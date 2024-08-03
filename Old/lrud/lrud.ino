#include <Servo.h>
Servo servo1;
Servo servo2; // Create a Servo object

int servoPosition1 = 0;
int servoPosition2 = 0;
int ledPin = 13;

void setup() {
  servo1.attach(9);
  servo2.attach(10); 
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);  // Initialize serial communication
}

void loop() {
  if (Serial.available() > 0) {
    char data = Serial.read();
    if (data == '1') {
      digitalWrite(ledPin, HIGH);  // Set the LED on
      servoPosition1 = 0;
    } else if (data == '0') {
      digitalWrite(ledPin, LOW);   // Set the LED off
      servoPosition1 = 90;
    }
     else if (data == '2') {
      digitalWrite(ledPin, LOW);   // Set the LED off
      servoPosition2 = 0;
    }
     else if (data == '3') {
      digitalWrite(ledPin,HIGH);   // Set the LED off
      servoPosition2 = 90;
    }
    
    servo1.write(servoPosition1);
    servo2.write(servoPosition2);
  
  }
  
  }
