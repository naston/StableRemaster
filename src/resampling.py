import numpy as np
from PIL import Image
import cv2

def get_new_window_padding(frame_shape):
    new_width = frame_shape[0]*16//9
    padding = (new_width-frame_shape[1])//2
    return padding

def get_new_window_mask(frame_shape):
    new_width = frame_shape[0]*16//9
    new_window_mask = np.ones((frame_shape[0],new_width))
    left_bar = (new_width-frame_shape[1])//2
    right_bar = new_width-left_bar
    new_window_mask[:,left_bar:right_bar]=0
    return new_window_mask

def get_sample_boxes(frame_shape, sample_shape):
    assert(sample_shape[1]<frame_shape[1])
    assert(sample_shape[0]<frame_shape[0])
    x_samples = np.ceil(frame_shape[1]/sample_shape[1]).astype(int)
    y_samples = np.ceil(frame_shape[0]/sample_shape[0]).astype(int)
    
    x_pix_overlap = x_samples*sample_shape[1]-frame_shape[1]
    y_pix_overlap = y_samples*sample_shape[0]-frame_shape[0]
    
    x_pix_offset = x_pix_overlap//(x_samples-1)
    y_pix_offset = y_pix_overlap//(y_samples-1)
    
    x_starts = np.zeros(x_samples)
    y_starts = np.zeros(y_samples)
    
    for i in range(1,len(x_starts)):
        x_starts[i]=i*sample_shape[1]-x_pix_offset
        if x_pix_offset!=x_pix_overlap:
            x_starts[i]-=1
            x_pix_overlap-=1
            
    for i in range(1,len(y_starts)):
        y_starts[i]=i*sample_shape[0]-y_pix_offset
        if y_pix_offset!=y_pix_overlap:
            y_starts[i]-=1
            y_pix_overlap-=1
    
    box_starts = np.array(np.meshgrid(y_starts, x_starts)).T.reshape(-1, 2).astype(int)
    
    return box_starts

def outpaint_frame(frame, mask, sample_width, sample_height, pipe):
    frame=frame.copy()
    mask=mask.copy()
    sample_starts = get_sample_boxes(frame.shape[0:2],[sample_height,sample_width])
    for (y,x) in sample_starts:
        sub_frame = frame[y:(y+sample_height),x:(x+sample_width),:]
        sub_mask = mask[y:(y+sample_height),x:(x+sample_width)]
        
        sub_frame = Image.fromarray(sub_frame).convert("RGB")
        sub_mask = Image.fromarray(sub_mask).convert("RGB")
        
        new_sub_frame = pipe(prompt='animated background',image=sub_frame, 
                             mask_image=sub_mask,height=sample_height,width=sample_width).images[0]
        frame[y:(y+sample_height),x:(x+sample_width),:] = np.array(new_sub_frame)
        mask[y:(y+sample_height),x:(x+sample_width)]=0
    return frame

def resample_frame(frame,mask,M, total_bg, total_mask, pipe):
    total_bg = total_bg.copy()
    total_mask = total_mask.copy()
    shape = frame.shape
    
    #need to pad total_bg and total_mask as well
    mask = (mask*-1+1).astype('uint8')
    p_frame, p_mask = get_pad_frame(frame,mask)
    _,inv_mask = get_pad_frame(frame,np.zeros(mask.shape))
    
    s_mask = p_mask.copy()
    s_mask[:,:]=1
    # transform frame and mask
    h, w = total_bg.shape[:2]
    h0,w0 = p_frame.shape[:2]

    t_mask = cv2.warpAffine(p_mask, M, (w,h))
    s_mask = cv2.warpAffine(s_mask,M,(w,h))

    # and mask w/ total_mask, then send to stable diffusion
    sd_mask = cv2.bitwise_and(t_mask , t_mask , mask = total_mask).astype('uint8')
    #send to stable diffusion w/ total_bg (only if the mask isn't all 0s?)
    
    #return t_frame,t_mask,sd_mask
    if np.any(sd_mask):
        sd_mask=sd_mask*255
        total_bg = outpaint_frame(total_bg, sd_mask, pipe=pipe)
        sd_mask=sd_mask//255
        sd_mask = (-1*sd_mask+1).astype('uint8')
        total_mask = cv2.bitwise_and(total_mask,sd_mask)
    
    # sample new total_bg using mask
    s_mask[np.where(s_mask>0)]=1
    sampled_bg = cv2.bitwise_and(total_bg , total_bg , mask = s_mask).astype('uint8')

    # reverse affine transformation
    M_inv = get_inv_trans(M)

    new_bg = cv2.warpAffine(sampled_bg.astype('uint8'), M_inv.astype('float64'), (w0,h0))
    idx = np.where(inv_mask>0)
    p_frame[idx]=new_bg[idx]

    # return frame alone
    return p_frame, total_bg, total_mask

def get_pad_frame(frame, mask):
    padding = get_new_window_padding(frame.shape)
    padded_frame = np.pad(frame, 
                             ((0,0),(padding,padding),(0,0)),
                             'constant',
                             constant_values=0)
    padded_mask = np.pad(mask, 
                             ((0,0),(padding,padding)),
                             'constant',
                             constant_values=1)
    return padded_frame, padded_mask

def prepare_total_resample(total_bg, total_mask, padding):
    total_mask = (total_mask*-1+1).astype('uint8')
    
    padded_bg = np.pad(total_bg, 
                             ((0,0),(padding,padding),(0,0)),
                             'constant',
                             constant_values=0)
    padded_mask = np.pad(total_mask, 
                             ((0,0),(padding,padding)),
                             'constant',
                             constant_values=1)
    return padded_bg, padded_mask

def get_inv_trans(M):
    R_inv = np.copy(M[:,:2]).T
    
    T_inv = np.zeros([2,1]) 
    T_inv[0,0] = -1*M[0,2]*M[0,0] - M[1,2]*M[0,1]
    T_inv[1,0] = -1*M[0,2]*M[0,0] + M[1,2]*M[0,1]
    
    M_inv = np.concatenate([R_inv,T_inv],axis=1)
    
    return M_inv