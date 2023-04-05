import os

import cv2
import numpy as np
from tqdm.auto import tqdm

DATA_DIR = "./data/02_intermediate/masked_bg/static_test1"

import cv2
import numpy as np

import cv2
import numpy as np

import cv2
import numpy as np


def seamless_clone(image1, image2):
    # Create a mask for the second image
    mask = np.zeros(image2.shape, dtype=image2.dtype)
    mask[:] = 255

    # Find the center point of the second image
    center = (image1.shape[1] // 2, image1.shape[0] // 2)

    # Blend the images using seamlessClone
    blended_image = cv2.seamlessClone(image2, image1, mask, center, cv2.NORMAL_CLONE)
    return blended_image


def stitch_images(image1, image2):
    # Detect features and keypoints using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Match the features using FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # Filter good matches using the ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 10:  # Minimum number of matches required for homography
        print("Not enough good matches found to stitch the images.")
        return image1

    # Extract the matched points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography using RANSAC
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)

    # Warp image2 using the computed homography
    h, w = image1.shape[:2]
    warped_image2 = cv2.warpPerspective(image2, H, (w, h))

    stitched_image = np.maximum(image1, warped_image2)

    return stitched_image


def stitch_multiple(images):
    if len(images) < 2:
        raise ValueError("At least two images are required for stitching.")

    stitched_image = images[0]
    for i in tqdm(range(1, len(images))):
        stitched_image = stitch_images(stitched_image, images[i])

    return stitched_image


if __name__ == "__main__":
    # Load the pre-masked images
    paths = [os.path.join(DATA_DIR, f"frame_{idx}.png") for idx in range(132)]
    images = [cv2.imread(path) for path in paths if os.path.exists(path)]
    images = [image for image in images if image is not None]
    # Mask the watermark
    height, width, _ = images[0].shape
    watermark_mask = np.ones((height, width), dtype=np.uint8) * 255
    watermark_mask[50:110, 785:935] = 0
    images = [cv2.bitwise_and(image, image, mask=watermark_mask) for image in images]

    # Stitch the images
    stitched_image = stitch_multiple(images)
    stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)

    # Display the result
    cv2.imshow("Stitched Image", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
