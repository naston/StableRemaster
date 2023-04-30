from diffusers import StableDiffusionInpaintPipeline
import torch

def get_sd_pipe():
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

    pipe = pipe.to(device)

    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy

    return pipe