from ultralytics import YOLO
import cv2
import time
import numpy as np
from collections import Counter

# Load YOLO model
model = YOLO('best.onnx')
cap = cv2.VideoCapture(0)

# Define class labels
class_labels = {
    0: "Paper",
    1: "Rock",
    2: "Scissors"
}

# UI Layout
screen_width, screen_height = 1280, 720
webcam_width = int(screen_width * 0.8)
ribbon_width = int(screen_width * 0.2)

# Game parameters
CHEAT_THRESHOLD = 0.03 # Minimum vertical movement (pixels)
PRE_ROUND_TIME = 3    # Time to show initial hands
HOLD_TIME = 2         # Time to capture final gesture
WIN_REQUIRED = 3      # Number of wins needed to win match

# Initialize game state
score = {"Left": 0, "Right": 0}
current_round = 1
game_active = True

def create_game_screen(frame):
    screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Webcam feed
    if frame is not None:
        resized_frame = cv2.resize(frame, (webcam_width, screen_height))
        screen[:, ribbon_width:ribbon_width+webcam_width] = resized_frame
    
    # Left ribbon
    cv2.rectangle(screen, (0, 0), (ribbon_width, screen_height), (40, 40, 40), -1)
    cv2.putText(screen, f"Round: {current_round}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(screen, f"Left: {score['Left']}", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(screen, f"Right: {score['Right']}", (20, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    return screen

def detect_hands(frame):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    
    hands = []
    for box, cls, conf in zip(boxes, classes, confidences):
        if conf > 0.6:  # Confidence threshold
            hands.append((box, cls))
    
    if len(hands) >= 2:
        sorted_hands = sorted(hands, key=lambda x: x[0][0])
        return sorted_hands[0], sorted_hands[1]
    return None, None

def validate_pre_round(left_hand, right_hand, prev_left_y, prev_right_y):
    l_box, l_cls = left_hand
    r_box, r_cls = right_hand
    
    # Check if both hands are showing Rock
    if l_cls != 1 or r_cls != 1:
        return False
    
    # Calculate box heights
    left_height = l_box[3] - l_box[1]
    right_height = r_box[3] - r_box[1]
    
    # Calculate movement percentages
    if prev_left_y is not None and prev_right_y is not None:
        left_move = abs(l_box[1] - prev_left_y) / left_height
        right_move = abs(r_box[1] - prev_right_y) / right_height
        return left_move > CHEAT_THRESHOLD and right_move > CHEAT_THRESHOLD
    
    return True  # First frame check

# Modify the draw_accurate_boxes function and its calls like this:

def draw_accurate_boxes(game_screen, left_hand, right_hand, width_scale, height_scale):
    """Draw bounding boxes with proper scaling and positioning"""
    # Draw left hand box
    if left_hand:
        l_box, l_cls = left_hand
        # Scale coordinates to resized webcam feed
        x1 = int(l_box[0] * width_scale) + ribbon_width
        y1 = int(l_box[1] * height_scale)
        x2 = int(l_box[2] * width_scale) + ribbon_width
        y2 = int(l_box[3] * height_scale)
        
        # Draw rectangle and label
        color = (0, 255, 0) if class_labels[l_cls] == "Rock" else (0, 0, 255)
        cv2.rectangle(game_screen, (x1, y1), (x2, y2), color, 3)
        cv2.putText(game_screen, class_labels[l_cls], 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Draw right hand box
    if right_hand:
        r_box, r_cls = right_hand
        # Scale coordinates to resized webcam feed
        x1 = int(r_box[0] * width_scale) + ribbon_width
        y1 = int(r_box[1] * height_scale)
        x2 = int(r_box[2] * width_scale) + ribbon_width
        y2 = int(r_box[3] * height_scale)
        
        # Draw rectangle and label
        color = (0, 255, 0) if class_labels[r_cls] == "Rock" else (0, 0, 255)
        cv2.rectangle(game_screen, (x1, y1), (x2, y2), color, 3)
        cv2.putText(game_screen, class_labels[r_cls], 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
validation_start = time.time()  # Captures the start time of validation

# In the validation loop, add scaling calculations:
while time.time() - validation_start < PRE_ROUND_TIME:
    ret, frame = cap.read()
    if not ret:
        continue
        
    # Calculate scaling factors
    original_height, original_width = frame.shape[:2]
    width_scale = webcam_width / original_width  # From original to resized width
    height_scale = screen_height / original_height  # From original to resized height
    
    game_screen = create_game_screen(frame)
    left_hand, right_hand = detect_hands(frame)
    
    # Pass scaling factors to drawing function
    draw_accurate_boxes(game_screen, left_hand, right_hand, width_scale, height_scale)
    
    # Rest of the validation loop remains the same...
        
        
        

def determine_winner(left, right):
    if left == right:
        return "Draw"
    wins = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
    return "Left" if wins[left] == right else "Right"

# Main game loop
while game_active:
    # Pre-round countdown
    game_screen = create_game_screen(None)
    start_time = time.time()
    
    while time.time() - start_time < PRE_ROUND_TIME:
        remaining = int(PRE_ROUND_TIME - (time.time() - start_time))
        blurred = cv2.GaussianBlur(game_screen, (99, 99), 30)
        cv2.putText(blurred, f"Next round in: {remaining}", (webcam_width//2-200, screen_height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow('Rock Paper Scissors', blurred)
        cv2.waitKey(1)
    
    # Pre-round validation with movement timer
# Initialize variables for tracking movement during pre-round validation
    validation_start = time.time()
    left_movements = []
    right_movements = []
    total_frames = 0
    prev_left_y = None
    prev_right_y = None
    

    while time.time() - validation_start < PRE_ROUND_TIME:
        ret, frame = cap.read()
        if not ret:
            continue

        game_screen = create_game_screen(frame)
        left_hand, right_hand = detect_hands(frame)

        # Calculate scaling factors
        original_height, original_width = frame.shape[:2]
        width_scale = webcam_width / original_width
        height_scale = screen_height / original_height

        # Draw bounding boxes
        draw_accurate_boxes(game_screen, left_hand, right_hand, width_scale, height_scale)

        # Movement tracking
        if left_hand and right_hand:
            l_box, _ = left_hand
            r_box, _ = right_hand
            current_left_y = l_box[1]
            current_right_y = r_box[1]

            if prev_left_y is not None and prev_right_y is not None:
                left_height = l_box[3] - l_box[1]
                right_height = r_box[3] - r_box[1]

                left_move = abs(current_left_y - prev_left_y) / left_height
                right_move = abs(current_right_y - prev_right_y) / right_height

                # Store movement percentages
                left_movements.append(left_move)
                right_movements.append(right_move)
                total_frames += 1
                
                cv2.putText(game_screen, f"L Move: {left_move:.2%}",
                                (ribbon_width + 50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0) if left_move >= CHEAT_THRESHOLD else (0, 0, 255), 2)
                cv2.putText(game_screen, f"R Move: {right_move:.2%}",
                                (ribbon_width + 50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0) if right_move >= CHEAT_THRESHOLD else (0, 0, 255), 2)

            # Update previous positions
            prev_left_y = current_left_y
            prev_right_y = current_right_y

        # Timer display
        remaining = PRE_ROUND_TIME - (time.time() - validation_start)
        cv2.putText(game_screen, f"Move rocks! {remaining:.1f}s",
                    (ribbon_width + 50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow('Rock Paper Scissors', game_screen)
        if cv2.waitKey(1) == 27:
            game_active = False
            break

    # Calculate average movement percentage
    if total_frames > 0:
        avg_left_move = sum(left_movements) / total_frames
        avg_right_move = sum(right_movements) / total_frames
    else:
        avg_left_move = avg_right_move = 0  # No hands detected

    print(f"Avg Left Move: {avg_left_move:.2%}, Avg Right Move: {avg_right_move:.2%}")
    

    # **Detect Cheating Based on Average Movement**
    if avg_left_move < CHEAT_THRESHOLD or avg_right_move < CHEAT_THRESHOLD:
        print("INVALID START! Average movement was too low.")
        if (avg_left_move < CHEAT_THRESHOLD and avg_right_move >= CHEAT_THRESHOLD):
            print("Left player cheated! Score reduced.")
            score["Left"] = max(0, score["Left"] - 1)
        elif (avg_left_move >= CHEAT_THRESHOLD and avg_right_move < CHEAT_THRESHOLD):
            print("Right player cheated! Score reduced.")
            score["Right"] = max(0, score["Right"] - 1)    
         
        cv2.putText(game_screen, "INVALID START!", (webcam_width//2-200, screen_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow('Rock Paper Scissors', game_screen)
        cv2.waitKey(1000)
        continue  # Restart the pre-round phase

        
    # validation_ratio = valid_frames / total_frames
    # print(f"Validation success rate: {validation_ratio:.2%}")
        
    # if validation_ratio < 0.7:
    #     cv2.putText(game_screen, "INVALID START!", (webcam_width//2-200, screen_height//2), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    #     cv2.imshow('Rock Paper Scissors', game_screen)
    #     cv2.waitKey(1000)
    #     continue
    
    # Capture final gestures
    gesture_start = time.time()
    left_gestures = []
    right_gestures = []
    
    while time.time() - gesture_start < HOLD_TIME:
        ret, frame = cap.read()
        if not ret:
            continue
            
        game_screen = create_game_screen(frame)
        left_hand, right_hand = detect_hands(frame)
        draw_accurate_boxes(game_screen, left_hand, right_hand , width_scale, height_scale)

        
        # Show hold timer
        remaining = HOLD_TIME - (time.time() - gesture_start)
        cv2.putText(game_screen, f"Hold position! {remaining:.1f}s", 
                   (ribbon_width + 100, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        if left_hand and right_hand:
            left_gestures.append(left_hand[1])
            right_gestures.append(right_hand[1])
        
        cv2.imshow('Rock Paper Scissors', game_screen)
        cv2.waitKey(1)
    
    # Determine winner
    if left_gestures and right_gestures:
        left_choice = Counter(left_gestures).most_common(1)[0][0]
        right_choice = Counter(right_gestures).most_common(1)[0][0]
        winner = determine_winner(class_labels[left_choice], class_labels[right_choice])
        
        # Update scores
        if winner == "Left":
            score["Left"] += 1
        elif winner == "Right":
            score["Right"] += 1
        
        # Show result
        result_text = f"{winner} Wins!" if winner != "Draw" else "Draw!"
        cv2.putText(game_screen, result_text, (webcam_width//2-150, screen_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow('Rock Paper Scissors', game_screen)
        cv2.waitKey(2000)
    
    current_round += 1
    
    # Check match winner
    if score["Left"] >= WIN_REQUIRED or score["Right"] >= WIN_REQUIRED:
        final_text = "LEFT PLAYER WINS!" if score["Left"] > score["Right"] else "RIGHT PLAYER WINS!"
        cv2.putText(game_screen, final_text, (webcam_width//2-250, screen_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow('Rock Paper Scissors', game_screen)
        cv2.waitKey(3000)
        game_active = False

cap.release()
cv2.destroyAllWindows()