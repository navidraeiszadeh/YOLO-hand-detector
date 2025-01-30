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
hold_time = 2  # Round duration in seconds
cheat_threshold = 0.7  # 10% movement threshold for cheat detection

# Track scores and rounds
left_hand_wins = 0
right_hand_wins = 0
round_count = 0

# Get N (number of rounds) from the user
while True:
    try:
        N = int(input("Enter an odd number of rounds (e.g., 3, 5, 7): "))
        if N % 2 == 1 and N > 0:
            break
        else:
            print("Please enter a valid odd number greater than 0.")
    except ValueError:
        print("Invalid input. Please enter an odd number.")

def determine_winner(choice1, choice2):
    """Determine the winner based on game rules."""
    if choice1 == choice2:
        return "Draw"
    if (choice1 == "Rock" and choice2 == "Scissors") or \
       (choice1 == "Paper" and choice2 == "Rock") or \
       (choice1 == "Scissors" and choice2 == "Paper"):
        return "Left Hand Wins"
    return "Right Hand Wins"

# Tracking previous positions to detect cheating
prev_left_box = None
prev_right_box = None
winner_text = ""

while round_count < N:
    hand1_results = []
    hand2_results = []
    cheat_detected = False

    ### 🎯 **Pre-round countdown timer (UX improvement)**
    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Set fullscreen mode
        cv2.namedWindow('YOLO Inference', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('YOLO Inference', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Display the countdown timer
        timer_text = f"Round {round_count + 1} starts in {countdown}..."
        cv2.putText(frame, timer_text, (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        cv2.imshow('YOLO Inference', frame)

        if cv2.waitKey(1000) == 27:  # 1000ms = 1 second
            cap.release()
            cv2.destroyAllWindows()
            exit()

    ### 🚀 **Start Round and Detect Hands**
    print(f"Starting Round {round_count + 1}/{N}")
    start_time = time.time()

    while True:
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
            # Sort boxes based on the x-coordinate to differentiate left and right hands
            sorted_indices = np.argsort([box[0] for box in boxes])
            left_box = boxes[sorted_indices[0]]
            right_box = boxes[sorted_indices[1]]

            left_hand_class = int(class_ids[sorted_indices[0]])
            right_hand_class = int(class_ids[sorted_indices[1]])

            hand1_results.append(left_hand_class)
            hand2_results.append(right_hand_class)

            # Draw bounding boxes and labels
            for idx, box in enumerate([left_box, right_box]):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_labels[int(class_ids[sorted_indices[idx]])]}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            ### ⚠️ **Cheat detection logic (after initial detection)**
            if prev_left_box is not None and prev_right_box is not None:
                left_movement = np.linalg.norm(np.array(left_box) - np.array(prev_left_box))
                right_movement = np.linalg.norm(np.array(right_box) - np.array(prev_right_box))

                box_width = abs(left_box[2] - left_box[0])
                if left_movement > cheat_threshold * box_width or right_movement > cheat_threshold * box_width:
                    cheat_detected = True
                    winner_text = "❌ Cheating Detected!"
                    print("❌ Cheating Detected!")

            # Save current positions for next iteration
            prev_left_box = left_box
            prev_right_box = right_box

        # 🎯 **Timer Display (UX improvement)**
        elapsed_time = int(time.time() - start_time)
        remaining_time = max(0, hold_time - elapsed_time)
        timer_text = f"⏳ {remaining_time}s - Round {round_count + 1}/{N}"
        cv2.putText(frame, timer_text, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

        ### ✅ **Determine Winner Only If Hands Are Detected**
        if remaining_time == 0 and not cheat_detected and hand1_results and hand2_results:
            left_final = class_labels[Counter(hand1_results).most_common(1)[0][0]]
            right_final = class_labels[Counter(hand2_results).most_common(1)[0][0]]

            print(f"Left Hand: {left_final}, Right Hand: {right_final}")
            round_winner = determine_winner(left_final, right_final)
            winner_text = round_winner
            if round_winner == "Left Hand Wins":
                left_hand_wins += 1
            elif round_winner == "Right Hand Wins":
                right_hand_wins += 1

            round_count += 1
            break

        cv2.putText(frame, winner_text, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)
        cv2.imshow('YOLO Inference', frame)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

### **Final Winner Announcement**
print("\n🎉 Game Over!")
print(f"🏆 Final Score - Left Hand: {left_hand_wins}, Right Hand: {right_hand_wins}")
final_winner = "🎉 Left Hand Wins!" if left_hand_wins > right_hand_wins else "🎉 Right Hand Wins!"
cv2.putText(frame, final_winner, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
cv2.imshow('YOLO Inference', frame)
cv2.waitKey(5000)

cap.release()
cv2.destroyAllWindows()
