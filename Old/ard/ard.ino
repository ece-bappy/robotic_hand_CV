#include <Servo.h>

Servo servo1;
Servo servo2;

void setup() {
  servo1.attach(9);  // Attach servo1 to pin 9
  servo2.attach(10); // Attach servo2 to pin 10
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readString();
    
    if (data == "upleft") {
      servo1.write(0);  // Set servo1 to 0 degrees
      servo2.write(0);  // Set servo2 to 0 degrees
    } else if (data == "downleft") {
      servo1.write(0);  // Set servo1 to 0 degrees
      servo2.write(100);  // Set servo2 to 100 degrees
    } else if (data == "upright") {
      servo1.write(100);  // Set servo1 to 100 degrees
      servo2.write(0);  // Set servo2 to 0 degrees
    } else if (data == "downright") {
      servo1.write(100);  // Set servo1 to 100 degrees
      servo2.write(100);  // Set servo2 to 100 degrees
    }
  }
}
