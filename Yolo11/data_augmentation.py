import cv2
import numpy as np
import os
import random

def flip_image(img, bbox, image_size):
    h, w = image_size
    img_flipped = cv2.flip(img, 1)  # Flip horizontally
    bbox_flipped = [1 - bbox[0] - bbox[2], bbox[1], bbox[2], bbox[3]]  # Update bbox
    return img_flipped, bbox_flipped

def rotate_image(img, bbox, image_size, angle=15):
    h, w = image_size
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (w, h))

    # Approximate bbox adjustment (not pixel-perfect for large angles)
    bbox_px = [
        int(bbox[0] * w),
        int(bbox[1] * h),
        int(bbox[2] * w),
        int(bbox[3] * h),
    ]
    x, y, bw, bh = bbox_px
    new_x, new_y = int(x - bw // 2), int(y - bh // 2)
    new_bw, new_bh = bw, bh
    bbox_rotated = [new_x / w, new_y / h, new_bw / w, new_bh / h]
    return img_rotated, bbox_rotated

def adjust_brightness(img, factor=1.2):
    img_adjusted = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    return img_adjusted

def augment_dataset(image_dir, label_dir, output_dir, image_size, augmentations=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    total_augmented = 0

    for image_file in image_files:
        # Load image and label
        img_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            label = f.readline().strip().split()
            class_name = label[0]
            bbox = list(map(float, label[1:]))  # [x, y, w, h]

        # Apply augmentations
        for i in range(augmentations):
            aug_img = img.copy()
            aug_bbox = bbox.copy()

            # Randomly choose augmentation
            aug_type = random.choice(['flip', 'rotate', 'brightness'])
            if aug_type == 'flip':
                aug_img, aug_bbox = flip_image(aug_img, aug_bbox, (h, w))
            elif aug_type == 'rotate':
                aug_img, aug_bbox = rotate_image(aug_img, aug_bbox, (h, w), angle=random.randint(-15, 15))
            elif aug_type == 'brightness':
                aug_img = adjust_brightness(aug_img, factor=random.uniform(0.8, 1.2))

            # Save augmented image and label
            aug_img_name = f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg"
            aug_label_name = f"{os.path.splitext(image_file)[0]}_aug_{i}.txt"

            cv2.imwrite(os.path.join(output_dir, aug_img_name), aug_img)
            with open(os.path.join(output_dir, aug_label_name), 'w') as f:
                f.write(f"{class_name} {' '.join(map(str, aug_bbox))}")

            total_augmented += 1

    print(f"Total augmented images created: {total_augmented}")

if __name__ == "__main__":
    # Input/output directories
    IMAGE_DIR = './images'  # Directory containing original images
    LABEL_DIR = './labels'  # Directory containing corresponding labels
    OUTPUT_DIR = './augmented_data'  # Directory to save augmented images and labels

    # Image size (512x512 as per your dataset)
    IMAGE_SIZE = (512, 512)

    # Number of augmentations per image
    AUGMENTATIONS = 5

    # Augment the dataset
    augment_dataset(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, IMAGE_SIZE, augmentations=AUGMENTATIONS)
