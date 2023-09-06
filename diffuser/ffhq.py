
from diffusers import DiffusionPipeline
import torch
import argparse
from torchvision.utils import make_grid, save_image
import os
import numpy as np
# setup args
parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--name', type=str, default=None)
args = parser.parse_args()

model_id = "google/ncsnpp-ffhq-1024"

# load model and scheduler
sde_ve = DiffusionPipeline.from_pretrained(model_id).to("cuda")

# run pipeline in inference (sample random noise and denoise)
#gen = torch.Generator(torch.device('cpu')).manual_seed(int(123))
image_list = []
seed_list = [i for i in range(1)]
for seed in seed_list:
    image = sde_ve(batch_size=1, seed=seed, output_type='tensor')
    print(image.shape)
    image_list.append(image)

images = torch.cat(image_list, dim=0)
#images_ = (images + 1) / 2.
images_ = images
print("len:", len(images))
image_grid = make_grid(images_, nrow=int(np.sqrt(len(images))))
save_image(image_grid, os.path.join(f'{args.name}.png'))

# # save image
# if args.name is not None:
#     image.save(f"{args.name}.png")
# else:
#     image.save("sde_ve_generated_image.png")

