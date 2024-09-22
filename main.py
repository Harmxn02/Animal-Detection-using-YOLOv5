import warnings
import torch
import cv2
from PIL import Image
import numpy as np
import yaml  # Import yaml for configuration

import subprocess

warnings.filterwarnings("ignore", category=FutureWarning)


# Function to train the custom model
def train_custom_model():
    # Path to the YOLOv5 repository
    yolo_path = "."  # Adjust if your path is different

    # Command to train the model
    command = [
        "python",
        "train.py",  # Call the training script
        "--data",
        "./animals-10.yaml",  # Path to your YAML config
        "--weights",
        "yolov5s.pt",  # Pre-trained weights
        "--epochs",
        "10",  # Number of epochs
        "--batch-size",
        "16",  # Batch size
        "--img-size",
        "640",  # Image size
    ]

    # Run the training process
    subprocess.run(command, cwd=yolo_path)

    print("Training complete!")


# OpenCV function to process video and detect animals
def detect_animals_in_video(video_path, output_path):

    # Load YOLOv5 model
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path="./runs/train/exp3/weights/best.pt"
    )

    # Load the video file
    cap = cv2.VideoCapture(video_path)

    # Get video dimensions and frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to save the output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to RGB (required by YOLOv5 model)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Perform detection
        results = model(img)

        # Convert results to a format suitable for OpenCV
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        # Draw boxes and labels on the frame
        n = len(labels)
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:  # Draw boxes for objects with confidence >= 50%
                x1, y1, x2, y2 = (
                    int(row[0] * width),
                    int(row[1] * height),
                    int(row[2] * width),
                    int(row[3] * height),
                )

                # Optionally apply a scaling factor
                scale_factor = 0.8
                x1, y1 = int(x1 * scale_factor), int(y1 * scale_factor)
                x2, y2 = int(x2 * scale_factor), int(y2 * scale_factor)

                confidence = row[4]

                # Draw a rectangle around the object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Get the label name and confidence score
                label_name = model.names[int(labels[i])]
                label = f"{label_name} {confidence:.2f}"

                # Put the label text above the rectangle
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Write the frame with the detections to the output video
        out.write(frame)

        # Show the frame (optional)
        cv2.imshow("Animal Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Main execution flow
if __name__ == "__main__":
    # Uncomment to train the model
    # train_custom_model()

    # Call the function to detect animals in the input video and save the output video
    detect_animals_in_video(
        "../video/input_video.mp4", "../video/output_video_custom_model.mp4"
    )


# ! BOUNDING BOXES ARE TOO BIG, I SUSPECT THIS IS BECAUSE THE TRAINING DATA HAD BOUNDING BOXES THAT WERE THE FULL SIZE OF THE IMAGE