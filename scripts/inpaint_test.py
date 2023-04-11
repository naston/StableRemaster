from diffusers import StableDiffusionInpaintPipeline
import torch
import numpy as np
from PIL import Image
import os

print('-'*30)
print(os.getcwd())
print('-'*30)

if torch.cuda.is_available():
    print('Device: CUDA')
    print('-'*30)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    device = torch.device('cuda')
else:
    print('Device: CPU')
    print('-'*30)
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",)
    device = torch.device('cpu')

pipe.to(device)

indir = './data/02_intermediate'
outdir = './data/03_final/'

#prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image_fp = f'{indir}/Lenna.png'
mask_fp = f'{indir}/Lenna_mask.png'

image = Image.open(image_fp).convert("RGB")
mask = np.zeros([512,512])
mask[:256,:]=255

print(np.array(image).shape)
print(np.array(mask).shape)

#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
im = pipe(prompt='',image=image, mask_image=mask).images[0]
im.save(f"{outdir}/Lenna_inpaint2.png")