import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Data directories and CSV file
data_dir = "D:/42/robot/hand/img"
csv_file = "data.csv"

# Load data from the CSV file
data = pd.read_csv(csv_file)
file_paths = data['file_path'].values
labels = data['label'].values

# Preprocess data
X = []  # Images
y = []  # Labels

for file_path, label in zip(file_paths, labels):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (64, 64))  # Resize to a consistent size
    X.append(image)
    y.append(label)

X = np.array(X)  # Convert to NumPy array
X = X / 255.0  # Normalize pixel values (0-255) to (0-1)
y = to_categorical(y, num_classes=2)  # Convert labels to one-hot encoded vectors

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Corrected line
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='softmax')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {accuracy:.2f}')
