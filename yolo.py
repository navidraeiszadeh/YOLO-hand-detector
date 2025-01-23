import time
import cv2
import numpy as np

class YOLO:
    def __init__(self, config, model, labels, size=416, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size
        self.labels = labels
        self.output_names = []
        
        try:
            self.net = cv2.dnn.readNetFromDarknet(config, model)
        except:
            raise ValueError("Couldn't find the models! Please check the model paths.")

        ln = self.net.getLayerNames()
        for i in self.net.getUnconnectedOutLayers():
            self.output_names.append(ln[int(i) - 1])

    def inference(self, image):
        ih, iw = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.output_names)
        end = time.time()
        inference_time = end - start

        boxes, confidences, classIDs = [], [], []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.confidence:
                    box = detection[0:4] * np.array([iw, ih, iw, ih])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = max(0, int(centerX - (width / 2)))
                    y = max(0, int(centerY - (height / 2)))
                    w, h = int(width), int(height)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        results, cropped_hands = [], []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                id = classIDs[i]
                confidence = confidences[i]

                results.append((id, self.labels[id], confidence, x, y, w, h))

                # Crop the detected hand from the image
                cropped_hand = image[y:y+h, x:x+w]

                # Preprocess the cropped hand for further processing
                preprocessed_hand = self.preprocess(cropped_hand)

                cropped_hands.append(preprocessed_hand)

        return iw, ih, inference_time, results, cropped_hands

    def preprocess(self, image):
        IMG_SIZE = (300, 300)
        print ("sasasasasasasasas")
        if image.size == 0:
            return None  # Handle empty crops safely

        image = cv2.resize(image, IMG_SIZE)  # Resize image to 300x300
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = image.astype('float32') / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=-1)  # Add channel dimension for compatibility
        return image
