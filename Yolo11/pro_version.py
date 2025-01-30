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

# Parameters
time_threshold = 0.5  # Process images every 0.5 seconds
cheat_threshold = 0.1  # 10% movement threshold for cheat detection

# Get N (number of wins required) from the user
while True:
    try:
        N = int(input("Enter the number of wins required (e.g., 3, 5, 7): "))
        if N > 0:
            break
        else:
            print("Please enter a valid number greater than 0.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Winner Tracking
left_hand_wins = 0
right_hand_wins = 0
round_count = 1  # Start from round 1

# Function to determine the winner
def determine_winner(choice1, choice2):
    if choice1 == choice2:
        return "DRAW"
    if (choice1 == "Rock" and choice2 == "Scissors") or \
       (choice1 == "Paper" and choice2 == "Rock") or \
       (choice1 == "Scissors" and choice2 == "Paper"):
        return "LEFT WINS"
    return "RIGHT WINS"

while left_hand_wins < N and right_hand_wins < N:
    hand1_results = []
    hand2_results = []
    cheat_detected = False
    hands_detected = False  # New flag to check if hands were detected

    ### 🎯 **Pre-round 3-second countdown**
    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Set fullscreen mode
        cv2.namedWindow('YOLO Inference', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('YOLO Inference', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Display the countdown timer
        timer_text = f"Round {round_count} starts in {countdown}..."
        cv2.putText(frame, timer_text, (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        cv2.imshow('YOLO Inference', frame)

        if cv2.waitKey(1000) == 27:  # Wait for 1 second
            cap.release()
            cv2.destroyAllWindows()
            exit()

    ### 🚀 **Start Detecting Hands (Only Proceeds When Hands Are Found)**
    print(f"🔍 Detecting hands for Round {round_count}...")
    
    while True:  # **Hold processing until hands are detected**
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break  

        results = model(frame)
        result = results[0]

        # Extract bounding boxes and class IDs
        boxes = result.boxes.xyxy
        class_ids = result.boxes.cls

        if len(boxes) >= 2:
            hands_detected = True  # Hands are detected
            sorted_indices = np.argsort([box[0] for box in boxes])
            left_hand_class = int(class_ids[sorted_indices[0]])
            right_hand_class = int(class_ids[sorted_indices[1]])

            hand1_results.append(left_hand_class)
            hand2_results.append(right_hand_class)

            # Draw bounding boxes and labels
            for idx, box in enumerate([boxes[sorted_indices[0]], boxes[sorted_indices[1]]]):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_labels[int(class_ids[sorted_indices[idx]])]}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show processing UI
        cv2.putText(frame, f"Processing...", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow('YOLO Inference', frame)

        if hands_detected:  # **Only break when hands are detected**
            break

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    ### ✅ **Determine Most Frequent Result**
    left_final = class_labels[Counter(hand1_results).most_common(1)[0][0]]
    right_final = class_labels[Counter(hand2_results).most_common(1)[0][0]]

    print(f"Left Hand: {left_final}, Right Hand: {right_final}")
    round_winner = determine_winner(left_final, right_final)

    ### 🎉 **Display the Winner with a Green Checkmark**
    if round_winner == "LEFT WINS":
        left_hand_wins += 1
        winner_text = "✅ LEFT WINS!"
        x1, y1, x2, y2 = map(int, boxes[sorted_indices[0]])
    elif round_winner == "RIGHT WINS":
        right_hand_wins += 1
        winner_text = "✅ RIGHT WINS!"
        x1, y1, x2, y2 = map(int, boxes[sorted_indices[1]])
    else:
        winner_text = "🤝 DRAW!"

    # Display the result
    cv2.putText(frame, winner_text, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)

    # Draw a green checkmark near the winner
    if round_winner != "DRAW":
        cv2.putText(frame, "✅", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)

    cv2.imshow('YOLO Inference', frame)
    cv2.waitKey(500)  # **Pause for 2 seconds before starting the next round**

    # ✅ **Only Increase Round Count if Hands Were Detected**
    if hands_detected:
        round_count += 1

### 🎉 **Final Winner Announcement**
print("\n🎉 Game Over!")
print(f"🏆 Final Score - Left: {left_hand_wins}, Right: {right_hand_wins}")

final_winner = "🎉 LEFT WINS THE GAME!" if left_hand_wins == N else "🎉 RIGHT WINS THE GAME!"
cv2.putText(frame, final_winner, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
cv2.imshow('YOLO Inference', frame)
cv2.waitKey(5000)

cap.release()
cv2.destroyAllWindows()
