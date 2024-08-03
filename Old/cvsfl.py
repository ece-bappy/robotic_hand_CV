import os
import csv

data_dir = "D:/42/robot/hand/img"  # Update to your data directory
csv_file = "data.csv"  # Name of the CSV file

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["file_path", "label"])  # Header row

    # For the "Stop" gesture images
    stop_dir = os.path.join(data_dir, "stop")
    for filename in os.listdir(stop_dir):
        if filename.endswith(".jpg"):
            writer.writerow([os.path.join(stop_dir, filename), "Stop"])

    # For the "Not Stop" images
    not_stop_dir = os.path.join(data_dir, "not_stop")
    for filename in os.listdir(not_stop_dir):
        if filename.endswith(".jpg"):
            writer.writerow([os.path.join(not_stop_dir, filename), "Not Stop"])
