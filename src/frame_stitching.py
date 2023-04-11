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

    if len(good_matches) < 4:  # Minimum number of matches required for homography
        print("Not enough good matches found to stitch the images.")
        print("# of good matches:", len(good_matches))
        return image1, mask1, np.array([[0,0,0],[0,0,0]])

    # Extract the matched points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the Affine Transformation using RANSAC
    M, _ = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC)
    M_o = np.copy(M)
    x_offset = M[0,2]
    y_offset = M[1,2]
    
    x_pad_size = int(np.floor(abs(x_offset)))
    y_pad_size = int(np.floor(abs(y_offset)))
    
    #if(abs(x_offset)>1 or abs(y_offset)>1):
        #print(y_offset,x_offset)
    
    # Warp image2 using the computed homography
    h, w = image1.shape[:2]
    
    h += y_pad_size
    w += x_pad_size
    
    if x_offset<0:
        #make affine transformation positive
        M[0,2]+=x_pad_size
    if y_offset<0:
        #make affine transformation positive
        M[1,2]+=y_pad_size
    #h = image1.shape[0]+image2.shape[0]
    #w = image1.shape[1]+image2.shape[1]
    warped_image2 = cv2.warpAffine(image2, M, (w, h))
    warped_mask = cv2.warpAffine(mask2,M, (w,h))
    
    #return M, warped_image2
    #return image2,warped_image2
    x_pad, y_pad = get_padding(x_offset,y_offset,x_pad_size, y_pad_size)
    
    padded_image1 = np.pad(image1, (y_pad,x_pad,(0,0)),'constant',constant_values=(0,0))
    padded_mask1 = np.pad(mask1, (y_pad,x_pad),'constant',constant_values=(0,0))
    
    #print(padded_mask1.shape,warped_mask.shape)
    #stitched_mask = np.any([mask1,warped_mask],axis=0).astype('uint8')
    stitched_mask = np.any([padded_mask1,warped_mask],axis=0).astype('uint8')
    
    masked_im1 = cv2.bitwise_and(padded_image1 , padded_image1 , mask = padded_mask1).astype('uint8')
    masked_im2 = cv2.bitwise_and(warped_image2 , warped_image2 , mask = warped_mask).astype('uint8')
    #masked_im1 = cv2.bitwise_and(image1 , image1 , mask = mask1).astype('uint8')
    #masked_im2 = cv2.bitwise_and(warped_image2 , warped_image2 , mask = warped_mask).astype('uint8')
    
    warped_mask = warped_mask/scale
    masked_im2 = masked_im2/scale
    
    mask_count = np.sum([padded_mask1,warped_mask],axis=0)
    #mask_count = np.sum([mask1,warped_mask],axis=0)
    mask_count[np.where(mask_count==0)]=1
    im_sum = np.sum([masked_im1,masked_im2],axis=0)
    
    averaged_image = (im_sum/np.expand_dims(mask_count, axis=-1)).astype('uint8')
    
    return averaged_image, stitched_mask, M_o


def stitch_multiple(images, masks):
    if len(images) < 2:
        raise ValueError("At least two images are required for stitching.")
    if len(images)!=len(masks):
        raise ValueError('Number of Masks and Images must match')
    stitched_image = images[0]
    stitched_mask = masks[0]
    Ms = [np.array([[1,0,0],[0,1,0]])]
    for i in tqdm(range(1, len(images))):
        stitched_image, stitched_mask, M = stitch_images(stitched_image, images[i], stitched_mask, masks[i],scale=i)
        #print(stitched_image.shape)
        stitched_image, stitched_mask = trim_black(stitched_image, stitched_mask)
        #print(stitched_image.shape)
        Ms.append(M)
    Ms = np.array(Ms)
    
    y_translate = np.min(Ms[:,1,2])
    x_translate = np.min(Ms[:,0,2])
    if y_translate<0:
        Ms[:,1,2]+=abs(y_translate)
    if x_translate<0:
        Ms[:,0,2]+=abs(x_translate)

    return stitched_image,stitched_mask, Ms

def trim_black(frame,mask):
    #print(frame.shape)
    if not np.any(frame):
        print('fuck')
        return frame,mask
    
    it_bottom = -1
    while not(np.any(frame[it_bottom,:,:])):
        it_bottom-=1
    
    it_top = 0
    while not(np.any(frame[it_top,:,:])):
        it_top+=1
    
    it_right = -1
    while not(np.any(frame[:,it_right,:])):
        it_right-=1
    
    it_left = 0
    while not(np.any(frame[:,it_left,:])):
        it_left+=1
    
    it_bottom = frame.shape[0]+it_bottom+1
    it_right = frame.shape[1]+it_right+1
    
    frame = frame[it_top:it_bottom,it_left:it_right,:]
    mask = mask[it_top:it_bottom,it_left:it_right]
    
    return frame,mask
    
def get_padding(x_offset,y_offset,x_pad_size,y_pad_size):
    x_pad=None
    y_pad=None
    
    if x_offset<=0:
        x_pad = (0,x_pad_size)
    else:
        x_pad = (x_pad_size,0)
    if y_offset<=0:
        y_pad = (0,y_pad_size)
    else:
        y_pad = (y_pad_size,0)

    return x_pad,y_pad