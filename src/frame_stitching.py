import cv2
import numpy as np
from tqdm import tqdm

def stitch_images(image1, image2, mask1, mask2, scale=1):
    # Detect features and keypoints using SIFT
    sift = cv2.SIFT_create()
    #display_masked_image(image1,mask1)
    keypoints1, descriptors1 = sift.detectAndCompute(image1, mask1)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, mask2)

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
        return image1, mask1, image1

    # Extract the matched points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography using RANSAC
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)

    # Warp image2 using the computed homography
    h, w = image1.shape[:2]
    warped_image2 = cv2.warpPerspective(image2, H, (w, h))
    warped_mask = cv2.warpPerspective(mask2,H, (w,h))

    stitched_mask = np.any([mask1,warped_mask],axis=0).astype('uint8')
    
    masked_im1 = cv2.bitwise_and(image1 , image1 , mask = mask1).astype('uint8')
    masked_im2 = cv2.bitwise_and(warped_image2 , warped_image2 , mask = warped_mask).astype('uint8')
    
    warped_mask = warped_mask/scale
    masked_im2 = masked_im2/scale
    
    mask_count = np.sum([mask1,warped_mask],axis=0)
    mask_count[np.where(mask_count==0)]=1
    im_sum = np.sum([masked_im1,masked_im2],axis=0)
    
    averaged_image = (im_sum/np.expand_dims(mask_count, axis=-1)).astype('uint8')
    
    return averaged_image, stitched_mask


def stitch_multiple(images, masks):
    if len(images) < 2:
        raise ValueError("At least two images are required for stitching.")

    stitched_image = images[0]
    stitched_mask = masks[0]
    
    for i in tqdm(range(1, len(images))):
        stitched_image, stitched_mask = stitch_images(stitched_image, images[i], stitched_mask, masks[i],scale=i)
    return stitched_image,stitched_mask