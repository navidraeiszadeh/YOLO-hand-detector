import cv2
import threading
import queue
import time
from yolo import YOLO

# Load YOLO model
yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

# Use CPU processing for compatibility and stability
yolo.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Global variables
frame_queue = queue.Queue()
processed_frame = None
lock = threading.Lock()
frame_skip = 2  # Process every 2nd frame to improve speed
padding = 40  # Increased padding to make bounding box much bigger
fps_counter = []

def process_frame():
    global processed_frame
    global processed_hands  # Store processed images for main thread display

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break

            start_time = time.time()
            with lock:
                width, height, inference_time, results, cropped_hands = yolo.inference(frame)

                processed_hands = []  # Clear the previous processed hands

                for i, (id, name, confidence, x, y, w, h) in enumerate(results):
                    # Expand bounding box with padding
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(width - x, w + 2 * padding)
                    h = min(height - y, h + 2 * padding)

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Save the processed cropped hand for display in the main thread
                    if cropped_hands[i] is not None:
                        processed_hand = cropped_hands[i].squeeze()  # Remove extra dimension
                        processed_hand_display = (processed_hand * 255).astype('uint8')
                        processed_hands.append(processed_hand_display)

                processed_frame = frame
                end_time = time.time()
                fps_counter.append(1 / (end_time - start_time))

# Start frame processing thread
processing_thread = threading.Thread(target=process_frame, daemon=True)
processing_thread.start()

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

processed_hands = []  # Initialize outside the loop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for better user experience
    frame_count += 1

    if frame_count % frame_skip == 0:
        frame_queue.put(frame.copy())  # Add frame to processing queue

    with lock:
        if processed_frame is not None:
            # Display processed frame with bounding boxes
            fps = sum(fps_counter[-10:]) / len(fps_counter[-10:]) if len(fps_counter) > 10 else 0
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Hand Detection", processed_frame)

            # Display all processed hands in separate windows
            for idx, hand_img in enumerate(processed_hands):
                hand_display = cv2.cvtColor(hand_img, cv2.COLOR_GRAY2BGR)
                cv2.imshow(f"Processed Hand {idx}", cv2.resize(hand_display, (300, 300)))

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        frame_queue.put(None)  # Signal processing thread to stop
        break

cap.release()
cv2.destroyAllWindows()

