from ultralytics import YOLO    
import cv2
import time
import numpy as np
from collections import Counter

# Load the YOLO model
model = YOLO('best.onnx')
cap = cv2.VideoCapture(0)

# Define class labels for better readability
class_labels = {
    0: "Paper",
    1: "Rock",
    2: "Scissors"
}

# Tracking parameters
hand1_results = []
hand2_results = []
hold_time = 2  # Time in seconds to collect data
time_threshold = 0.5  # Process images every 0.5 seconds
detection_active = True  # Flag to control detection processing

start_time = time.time()
last_frame = None  # To store the last processed frame

def determine_winner(choice1, choice2):
    """Determine the winner based on game rules."""
    if choice1 == choice2:
        return "Draw"
    if (choice1 == "Rock" and choice2 == "Scissors") or \
       (choice1 == "Paper" and choice2 == "Rock") or \
       (choice1 == "Scissors" and choice2 == "Paper"):
        return "Hand 1 Wins"
    return "Hand 2 Wins"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break  

    if detection_active:
        # Process the image at a fixed interval
        # time.sleep(time_threshold)

        results = model(frame)
        result = results[0]

        # Extract bounding boxes, confidence scores, and class IDs
        boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
        class_ids = result.boxes.cls  # Class IDs

        # Ensure at least two hands are detected
        if len(boxes) >= 2:
            # Sort boxes based on the x-coordinate to differentiate left and right hands
            sorted_indices = np.argsort([box[0] for box in boxes])
            
            # Get the first hand (left) and second hand (right)
            hand1_box = boxes[sorted_indices[0]]
            hand2_box = boxes[sorted_indices[1]]

            hand1_class = int(class_ids[sorted_indices[0]])
            hand2_class = int(class_ids[sorted_indices[1]])

            hand1_results.append(hand1_class)
            hand2_results.append(hand2_class)

            # Draw bounding boxes
            for idx, box in enumerate([hand1_box, hand2_box]):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_labels[int(class_ids[sorted_indices[idx]])]}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            last_frame = frame.copy()  # Store the last processed frame

        # Check if enough time has passed for a decision
        if time.time() - start_time >= hold_time:
            if hand1_results and hand2_results:
                # Determine the most frequent prediction for each hand
                hand1_final = class_labels[Counter(hand1_results).most_common(1)[0][0]]
                hand2_final = class_labels[Counter(hand2_results).most_common(1)[0][0]]

                print(f"Final Hand 1: {hand1_final}, Final Hand 2: {hand2_final}")
                winner = determine_winner(hand1_final, hand2_final)
                print(f"Result: {winner}")

                # Stop processing new frames but keep showing the last detected frame
                detection_active = False  

    # Show the last processed frame if detection is stopped
    if last_frame is not None:
        cv2.imshow('YOLO Inference', last_frame)
    else:
        cv2.imshow('YOLO Inference', frame)

    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
