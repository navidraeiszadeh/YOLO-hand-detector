import cv2
import os
from os import sep
import numpy as np
import mediapipe as mp


def find_hand(RGBimg, results, margin=20):
    if(results.multi_hand_landmarks):
        xList = []
        yList = []
        lmList = []
        myHand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = RGBimg.shape
            px, py = int(lm.x * w), int(lm.y * h)
            xList.append(px)
            yList.append(py)
            lmList.append([px, py])
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax - xmin, ymax - ymin
        return xmin - margin, ymin - margin, boxW + 2*margin, boxH + 2*margin
    return None


class CamDataset:
    def __init__(
            self, images_path, labels_path, image_dsize, class_name, image_format='jpg'):
        self.class_name = class_name
        self.image_dsize = image_dsize
        self.image_format = image_format
        self.labels_path = labels_path
        self.images_path = images_path

    def start_streaming(self):
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(static_image_mode=False,
                              max_num_hands=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
        img_counter = 0
        cam = cv2.VideoCapture(1) 
        # video capture must be zero for original webcam of laptap or 1 for other webcams

        print("enter 'q' to quit\nenter 's' to save data")
        while True:
            _, img = cam.read()
            img = cv2.resize(img, self.image_dsize)
            RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(RGBimg)
            hand = find_hand(RGBimg, results)

            if(hand):
                x, y, w, h = hand
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if(cv2.waitKey(1) & 0xFF == ord("s")):
                    image_path = "{0}{1}{2}_{3}.{4}".format(
                        self.images_path,
                        sep,
                        self.class_name,
                        img_counter,
                        self.image_format,)
                    annotation_path = "{0}{1}{2}_{3}.{4}".format(
                        self.labels_path,
                        sep,
                        self.class_name,
                        img_counter,
                        "txt",)
                    annotation_data = "{0} {1} {2} {3} {4}".format(
                        self.class_name,
                        x/self.image_dsize[0],
                        y/self.image_dsize[1],
                        w/self.image_dsize[0],
                        h/self.image_dsize[1])
                    cv2.imwrite(image_path, img)
                    annotation_file = open(annotation_path, "w")
                    annotation_file.write(annotation_data)
                    annotation_file.close()
                    print("image: '{0}' annotation: '{1}'".format(
                        image_path, annotation_path))
                    img_counter += 1
            elif(cv2.waitKey(1) & 0xFF == ord("s")):
                print("no objects")
            cv2.imshow("webcam stream", cv2.resize(img, (800, 600)))
            if(cv2.waitKey(1) & 0xFF == ord("q")):
                break


if __name__ == "__main__":
    CLASS_NAME = "Scissors"  #save scissors image & Rock images & Paper images for all dataset
    IMAGE_SIZE = (512, 512)

    if not os.path.isdir('images'):
        os.system('mkdir images')

    if not os.path.isdir('labels'):
        os.system('mkdir labels')

    FILE_PATH = '.' + sep + 'images'
    LABELS_PATH = '.' + sep + 'labels'

    cam = CamDataset(FILE_PATH, LABELS_PATH, IMAGE_SIZE, CLASS_NAME)
    cam.start_streaming()
