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

# Function to determine the winner based on game rules
def determine_winner(choice1, choice2):
    if choice1 == choice2:
        return "Draw"
    if (choice1 == "Rock" and choice2 == "Scissors") or \
       (choice1 == "Paper" and choice2 == "Rock") or \
       (choice1 == "Scissors" and choice2 == "Paper"):
        return "Left Hand Wins"
    return "Right Hand Wins"

# Get N (number of rounds) from the user (ensure it's odd)
while True:
    try:
        N = int(input("Enter an odd number of rounds (e.g., 3, 5, 7): "))
        if N % 2 == 1 and N > 0:
            break
        else:
            print("Please enter a valid odd number greater than 0.")
    except ValueError:
        print("Invalid input. Please enter an odd number.")

# Tracking parameters
time_threshold = 0.5  # Process images every 0.5 seconds
hold_time = 3  # Time in seconds to collect data per round
left_hand_wins = 0
right_hand_wins = 0
round_count = 0
detection_active = True  # Flag to control game loop
winner_text = ""  # To store the winner result text for display

while round_count < N:
    hand1_results = []
    hand2_results = []
    start_time = time.time()

    print(f"Round {round_count + 1}/{N} - Place your hands!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break  

        # time.sleep(time_threshold)

        results = model(frame)
        result = results[0]

        # Extract bounding boxes and class IDs
        boxes = result.boxes.xyxy
        class_ids = result.boxes.cls

        if len(boxes) >= 2:
            # Sort boxes based on the x-coordinate to differentiate left and right hands
            sorted_indices = np.argsort([box[0] for box in boxes])
            
            # Get the left and right hands
            left_hand_class = int(class_ids[sorted_indices[0]])  # Left hand (smaller x value)
            right_hand_class = int(class_ids[sorted_indices[1]])  # Right hand (larger x value)

            hand1_results.append(left_hand_class)
            hand2_results.append(right_hand_class)

            # Draw bounding boxes and labels
            for idx, box in enumerate([boxes[sorted_indices[0]], boxes[sorted_indices[1]]]):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_labels[int(class_ids[sorted_indices[idx]])]}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Check if enough time has passed to process the round
        if time.time() - start_time >= hold_time:
            if hand1_results and hand2_results:
                left_final = class_labels[Counter(hand1_results).most_common(1)[0][0]]
                right_final = class_labels[Counter(hand2_results).most_common(1)[0][0]]

                print(f"Left Hand: {left_final}, Right Hand: {right_final}")
                round_winner = determine_winner(left_final, right_final)

                # If both hands have the same result, add 2 more rounds to N
                if round_winner == "Draw":
                    print("Draw detected! Adding 2 extra rounds.")
                    N += 2
                    winner_text = "Draw - Extra Rounds Added"
                else:
                    if round_winner == "Left Hand Wins":
                        left_hand_wins += 1
                    elif round_winner == "Right Hand Wins":
                        right_hand_wins += 1
                    
                    winner_text = round_winner

                print(f"Round Winner: {round_winner}")
                round_count += 1
                break  # Move to the next round

        # Show the webcam feed with winner text
        if winner_text:
            cv2.putText(frame, winner_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('YOLO Inference', frame)

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            detection_active = False
            break

# Determine overall winner
print("\nGame Over!")
print(f"Final Score - Left Hand: {left_hand_wins}, Right Hand: {right_hand_wins}")

if left_hand_wins > right_hand_wins:
    print("Overall Winner: Left Hand")
    winner_text = "Overall Winner: Left Hand"
elif right_hand_wins > left_hand_wins:
    print("Overall Winner: Right Hand")
    winner_text = "Overall Winner: Right Hand"
else:
    print("It's a Draw!")
    winner_text = "Overall Draw"

# Keep the webcam open to show final result
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, winner_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('YOLO Inference', frame)

    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
