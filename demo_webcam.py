import argparse
import cv2
from yolo import YOLO

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-d', '--device', type=int, default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for YOLO')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for YOLO')
ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
args = ap.parse_args()

# Load YOLO modela
if args.network == "normal":
    print("Loading YOLO model...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("Loading YOLO-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("Loading YOLOv4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("Loading YOLO-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("Starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(args.device)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    width, height, inference_time, results, cropped_hands = yolo.inference(frame)

    # Display FPS
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for i, (id, name, confidence, x, y, w, h) in enumerate(results):
        # Draw bounding box
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{name} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display cropped and preprocessed hand
        if cropped_hands[i] is not None:
            cv2.imshow(f"Processed Hand {i}", cv2.resize(cropped_hands[i], (150, 150)))

    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    if cv2.waitKey(20) == 27:  # Exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
