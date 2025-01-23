import cv2
import threading
import queue
import time
import gc
from yolo import YOLO
import tensorflow as tf
import numpy as np

# Load the trained classification model
model = tf.keras.models.load_model("models/my_model.h5")

# Define class labels for your model (modify accordingly)
class_labels = ["Class 0", "Class 1", "Class 2"]  # Update with your actual class names


# Load YOLO model (using tiny version for faster inference)
yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

# Use optimized OpenCV DNN backend (CPU or GPU if available)
yolo.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Global variables
frame_queue = queue.Queue()
processed_frame = None
processed_hands = []  # Store processed hand images for display
lock = threading.Lock()
frame_skip = 3  # Process every 3rd frame for better performance
padding = 40  # Padding to increase bounding box size
fps_counter = []

# Video capture settings for performance boost
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce frame latency

# Function to process the frame
def process_frame():
    global processed_frame
    global processed_hands

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break

            start_time = time.time()
            with lock:
                width, height, inference_time, results, cropped_hands = yolo.inference(frame)

                processed_hands = []  # Clear previous processed hands

                for i, (id, name, confidence, x, y, w, h) in enumerate(results):
                    # Expand the bounding box
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(width - x, w + 2 * padding)
                    h = min(height - y, h + 2 * padding)

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Process the cropped hand for recognition
                    if cropped_hands[i] is not None:
                        processed_hand = cropped_hands[i].squeeze()  # Remove extra dimension
                        processed_hand = (processed_hand * 255).astype('uint8')  # Convert back to uint8

                        # Resize to model input size
                        processed_hand = cv2.resize(processed_hand, (300, 300))
                        processed_hand = processed_hand.astype('float32') / 255.0  # Normalize
                        processed_hand = np.expand_dims(processed_hand, axis=[0, -1])  # Add batch and channel dims

                        # Predict using the model
                        predictions = model.predict(processed_hand)
                        predicted_class = np.argmax(predictions)
                        predicted_label = class_labels[predicted_class]
                        predicted_confidence = np.max(predictions)

                        # Print predicted output
                        print(f"Predicted: {predicted_label} with confidence: {predicted_confidence:.2f}")

                        # Display predicted label on the hand image
                        cv2.putText(frame, f"{predicted_label} ({predicted_confidence:.2f})", 
                                    (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0, 255, 0), 2)

                        processed_hands.append(processed_hand)

                processed_frame = frame
                end_time = time.time()
                fps_counter.append(1 / (end_time - start_time))
                gc.collect()


# Start frame processing in a separate thread
processing_thread = threading.Thread(target=process_frame, daemon=True)
processing_thread.start()

frame_count = 0

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
            fps = sum(fps_counter[-10:]) / len(fps_counter[-10:]) if len(fps_counter) > 10 else 0
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Hand Detection", processed_frame)

            # Display processed hand images with predictions
            for idx, hand_img in enumerate(processed_hands):
                hand_display = cv2.cvtColor((hand_img.squeeze() * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
                cv2.imshow(f"Processed Hand {idx}", cv2.resize(hand_display, (300, 300)))


    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        frame_queue.put(None)  # Signal processing thread to stop
        break

cap.release()
cv2.destroyAllWindows()
