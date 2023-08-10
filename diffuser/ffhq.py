
from diffusers import DiffusionPipeline
import torch
import argparse
# setup args
parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--name', type=str, default=None)
args = parser.parse_args()

model_id = "google/ncsnpp-ffhq-1024"

# load model and scheduler
sde_ve = DiffusionPipeline.from_pretrained(model_id).to("cuda")

# run pipeline in inference (sample random noise and denoise)
#gen = torch.Generator(torch.device('cpu')).manual_seed(int(123))
image = sde_ve().images[0]


# save image
if args.name is not None:
    image.save(f"{args.name}.png")
else:
    image.save("sde_ve_generated_image.png")

