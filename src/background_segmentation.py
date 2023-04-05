import cv2
import numpy as np

import torch
import torchvision
from torchvision import transforms as T

def background_segmentation_loader():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights = "DEFAULT")
    model.eval()

    transform = T.ToTensor()

    watermark_mask = np.ones((720,940)).astype('uint8')
    watermark_mask[50:110,785:935]=0
    
    def create_background_mask(frame, mask_conf, cat_conf):
        frame_input = transform(frame)
        with torch.no_grad():
            pred = model([frame_input])

        masks = pred[0]["masks"]
        labels = pred[0]['labels']
        scores = pred[0]['scores']

        background_mask = np.ones(masks[0,0].shape).astype("uint8")
        for i in np.where(scores>cat_conf)[0]:
            inv_obj_mask = (masks[i,0]<=mask_conf).numpy().astype("uint8")
            background_mask = np.all([background_mask,inv_obj_mask],axis=0) 

        return background_mask

    def get_background(frame, mask_conf=0.4, cat_conf=0.7):
        mask = create_background_mask(frame, mask_conf, cat_conf).astype('uint8')
        mask = cv2.bitwise_and(mask , mask , mask = watermark_mask).astype('uint8')
        bg = cv2.bitwise_and(frame , frame , mask = mask)
        return bg, mask
    
    return get_background