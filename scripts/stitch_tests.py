import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

DATA_DIR = "./data/02_intermediate/"


def stitch_images(image1, image2):
    # Detect features and keypoints using AKAZE
    akaze = cv2.AKAZE_create()
    keypoints1, descriptors1 = akaze.detectAndCompute(image1, None)
    keypoints2, descriptors2 = akaze.detectAndCompute(image2, None)

    # Match the features using FLANN-based matcher
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Filter good matches using the ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Calculate the average translation
    dx_total, dy_total = 0, 0
    for m in good_matches:
        dx_total += keypoints1[m.queryIdx].pt[0] - keypoints2[m.trainIdx].pt[0]
        dy_total += keypoints1[m.queryIdx].pt[1] - keypoints2[m.trainIdx].pt[1]

    dx_avg = int(dx_total / len(good_matches))
    dy_avg = int(dy_total / len(good_matches))

    # Shift image1 by the average translation
    h, w = image1.shape[:2]
    M = np.float32([[1, 0, dx_avg], [0, 1, dy_avg]])
    shifted_image1 = cv2.warpAffine(image1, M, (w, h))

    # Blend the images
    stitched_image = np.maximum(shifted_image1, image2)

    return stitched_image

if __name__ == "__main__":
    # Load the pre-masked images
    frame1, frame2 = os.path.join(DATA_DIR, "frame1.png"), os.path.join(DATA_DIR, "frame2.png")
    image1 = cv2.imread(frame1)
    image2 = cv2.imread(frame2)

    # Stitch the images
    stitched_image = stitch_images(image1, image2)

    # Convert images from BGR to RGB for correct display in matplotlib
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)

    # Display the input images and the stitched image using matplotlib
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image1_rgb)
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image2_rgb)
    plt.title("Image 2")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(stitched_image_rgb)
    plt.title("Stitched Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()