# StableRemaster
The way we perceive content has been revolutionized by the rapid progress of modern displays. From stunning 4K nature documentaries to sports broadcasts that provide such clarity that anyone can act as a referee, modern displays have enhanced the viewing experience. However, older content developed for CRT or early Plasma screen TVs has become outdated quickly and no longer meets current aspect ratio and resolution standards, resulting in a less enjoyable re-watching experience of beloved shows. Fortunately, we can solve this problem with the use of diffusion models to adapt old content to meet contemporary expectations.

Although Stable Diffusion\cite{https://doi.org/10.48550/arxiv.2112.10752} has gained popularity for image generation, its application to video often faces a challenge of temporal continuity. This poses a significant issue for aspect ratio expansion as the expanded content typically comprises static backgrounds. Our proposed project aims to address this limitation by utilizing the static spatial bias to govern the generation of new content and ensure that multiple instances of the same background region are not produced. By doing so, we can overcome the issue of temporal continuity in video generated using Stable Diffusion, thereby enhancing the quality and coherence of the output.

To achieve this goal, we will utilize a novel approach that combines Stable Diffusion with machine learning techniques. Specifically, we will use a machine learning model to identify and extract the static background regions from the input video. These regions will then be used as a reference to generate new content that preserves the temporal coherence of the video. Additionally, we will explore the use of other techniques such as motion estimation to further improve the quality of the output. 

## Pipeline
- Scene Segmentation
- Object Masking?
- Background Merging
- Sampled Outpainting
- Sharpening? (Post Processing)

## Setup
```
git clone https://github.com/naston/StableRemaster
cd StableRemaster
conda env create -f environment.yml
```
Download ffmpeg from [https://ffmpeg.org/download.html]
Verify download in terminal using: `ffmpeg`

## Developement
- Create your own branch of the name `<initials>_<task-name>`
- A branch task should be small in scope and able to be completed within a week
- After the task is complete create a PR and assign all team members as reviewers
- Once branch passes PR it will be merged and deleted
