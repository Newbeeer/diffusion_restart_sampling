# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, SDEScheduler, EulerDiscreteScheduler
import torch
import argparse
import os
from torchvision.utils import make_grid, save_image
import numpy as np

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=50, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--num', type=int, default=1)
parser.add_argument('--w', type=float, default=7.5)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum geneated samples')
parser.add_argument('--save_path', type=str, default='./generated_images')
parser.add_argument('--scheduler', type=str, default='DDPM')
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument('--second', action='store_true', default=False, help='second order ODE')
parser.add_argument('--sigma', action='store_true', default=False, help='use sigma')
args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))


prompt_list = ["a photo of an astronaut riding a horse on mars", "a raccoon playing table tennis",
          "Intricate origami of a fox in a snowy forest", "A transparent sculpture of a duck made out of glass"]

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
    elif args.scheduler == 'SDE':
        pipe.scheduler = SDEScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == 'ODE':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.use_karras_sigmas = False
    else:
        raise NotImplementedError

    prompt = [prompt_] * 16
    # DDIM
    generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)
    args.steps = 100
    image = pipe(prompt, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w,
                 restart=args.restart, second_order=args.second, output_type='tensor').images
    image_grid = make_grid(torch.from_numpy(image).permute(0, 3, 1, 2), nrow=int(np.sqrt(len(image))))
    save_image(image_grid,
               f"./vis/{prompt_}_{args.scheduler}_restart_{args.restart}_steps_{args.steps}_w_{args.w}_seed_{args.generate_seed}.png")

    # Heun
    generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)
    args.steps = 51
    image = pipe(prompt, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w,
                 restart=args.restart, second_order=True, output_type='tensor').images
    image_grid = make_grid(torch.from_numpy(image).permute(0, 3, 1, 2), nrow=int(np.sqrt(len(image))))
    save_image(image_grid,
               f"./vis/{prompt_}_{args.scheduler}_restart_{args.restart}_steps_{args.steps}_w_{args.w}_seed_{args.generate_seed}_second.png")

    # Restart
    generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)
    args.steps = 30
    image = pipe(prompt, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w, restart=True,
                 second_order=args.second, output_type='tensor').images
    image_grid = make_grid(torch.from_numpy(image).permute(0, 3, 1, 2), nrow=int(np.sqrt(len(image))))
    save_image(image_grid,
               f"./vis/{prompt_}_{args.scheduler}_restart_True_steps_{args.steps}_w_{args.w}_seed_{args.generate_seed}.png")

    # DDPM
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)
    args.steps = 100
    image = pipe(prompt, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w,
                 restart=args.restart, output_type='tensor').images
    image_grid = make_grid(torch.from_numpy(image).permute(0, 3, 1, 2), nrow=int(np.sqrt(len(image))))
    save_image(image_grid,
               f"./vis/{prompt_}_DDPM_restart_{args.restart}_steps_{args.steps}_w_{args.w}_seed_{args.generate_seed}.png")

