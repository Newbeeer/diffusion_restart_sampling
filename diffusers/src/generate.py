# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, SDEScheduler, EulerDiscreteScheduler
import torch
from coco_data_loader import text_image_pair
from PIL import Image
import os
import pandas as pd
import argparse
import torch.nn as nn
from torch_utils import distributed as dist
import numpy as np
import tqdm

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py --steps 30 --scheduler DDIM --save_path /mnt/nfs_folder/imgs --bs 12
parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=50, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--w', type=float, default=7.5)
parser.add_argument('--s_noise', type=float, default=1.)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum geneated samples')
parser.add_argument('--save_path', type=str, default='./generated_images')
parser.add_argument('--scheduler', type=str, default='DDPM')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument('--second', action='store_true', default=False, help='second order ODE')
parser.add_argument('--sigma', action='store_true', default=False, help='use sigma')
args = parser.parse_args()


dist.init()

dist.print0('Args:')
for k, v in sorted(vars(args).items()):
    dist.print0('\t{}: {}'.format(k, v))
# define dataset / data_loader
# text2img_dataset = text_image_pair(dir_path='/scratch/ylxu/datasets/coco/subset', csv_path='/scratch/ylxu/datasets/coco/subset.csv')
# text2img_loader = torch.utils.data.DataLoader(dataset=text2img_dataset, batch_size=args.bs, shuffle=False)

df = pd.read_csv('/mnt/nfs_folder/coco/coco/subset.csv')
all_text = list(df['caption'])
all_text = all_text[: args.max_cnt]
#print("all text:", all_text)

num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
all_batches = np.array_split(np.array(all_text), num_batches)
rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]


index_list = np.arange(len(all_text))
all_batches_index = np.array_split(index_list, num_batches)
rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]


##### load stable diffusion models #####
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
dist.print0("default scheduler config:")
dist.print0(pipe.scheduler.config)
# print(pipe.scheduler.compatibles)
# print(pipe.scheduler.config)
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

generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)

##### setup save configuration #######
if args.name is None:
    save_dir = os.path.join(args.save_path,
                            f'scheduler_{args.scheduler}_steps_{args.steps}_restart_{args.restart}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_sigma_{args.sigma}')
else:
    save_dir = os.path.join(args.save_path,
                            f'scheduler_{args.scheduler}_steps_{args.steps}_restart_{args.restart}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_sigma_{args.sigma}_name_{args.name}')
if dist.get_rank() == 0 and not os.path.exists(save_dir):
    os.mkdir(save_dir)

## generate images ##
for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
    torch.distributed.barrier()
    text = list(mini_batch)
    #dist.print0(text)
    image = pipe(text, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w, restart=args.restart, second_order=args.second, dist=dist, S_noise=args.s_noise).images

    for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
        image[text_idx].save(os.path.join(save_dir, f'{global_idx}.png'))

# Done.
torch.distributed.barrier()
if dist.get_rank() == 0:
    d = {'caption': all_text}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(save_dir, 'subset.csv'))