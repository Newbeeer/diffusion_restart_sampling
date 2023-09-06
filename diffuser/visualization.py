# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, SDEScheduler, EulerDiscreteScheduler
import torch
import argparse
import os
from torchvision.utils import make_grid, save_image
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=30, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--w', type=float, default=8)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum geneated samples')
parser.add_argument('--save_path', type=str, default='./generated_images')
parser.add_argument('--scheduler', type=str, default='DDIM')
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument('--second', action='store_true', default=False, help='second order ODE')
parser.add_argument('--sigma', action='store_true', default=False, help='use sigma')
parser.add_argument('--prompt', type=str, default='a photo of an astronaut riding a horse on mars')

args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

os.makedirs('./vis', exist_ok=True)

# prompt_list = ["a photo of an astronaut riding a horse on mars", "a raccoon playing table tennis",
#           "Intricate origami of a fox in a snowy forest", "A transparent sculpture of a duck made out of glass"]
prompt_list = [args.prompt]


for prompt_ in prompt_list:

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    print("default scheduler config:")
    print(pipe.scheduler.config)

    pipe = pipe.to("cuda")

    if args.scheduler == 'DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == 'DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.use_sigma = args.sigma
    else:
        raise NotImplementedError

    prompt = [prompt_] * 16

    # Restart
    generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)
    out, _ = pipe(prompt, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w,
                 restart=args.restart, second_order=args.second, output_type='tensor')
    image = out.images

    print(f'image saving to ./vis/{prompt_}_{args.scheduler}_restart_True_steps_{args.steps}_w_{args.w}_seed_{args.generate_seed}.png')
    image_grid = make_grid(torch.from_numpy(image).permute(0, 3, 1, 2), nrow=int(np.sqrt(len(image))))
    save_image(image_grid,
               f"./vis/{prompt_}_{args.scheduler}_restart_True_steps_{args.steps}_w_{args.w}_seed_{args.generate_seed}.png")
