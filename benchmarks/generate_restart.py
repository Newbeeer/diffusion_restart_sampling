# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch.distributions import Beta
import glob
from torch_utils import misc
#----------------------------------------------------------------------------
# Proposed Restart sampler

def restart_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=0,
    pfgmpp=False, restart_info="", restart_gamma=0
):

    def get_steps(min_t, max_t, num_steps, rho):

         step_indices = torch.arange(num_steps, dtype=torch.float, device=latents.device)
         t_steps = (max_t ** (1 / rho) + step_indices / (num_steps - 1) * (min_t ** (1 / rho) - max_t ** (1 / rho))) ** rho
         return t_steps

    N = net.img_channels * net.img_resolution * net.img_resolution
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0
    total_step = len(t_steps)
    if pfgmpp:
        x_next = latents.to(torch.float64)
    else:
        x_next = latents.to(torch.float64) * t_steps[0]
        # Main sampling loop.

    # {[num_steps, number of restart iteration (K), t_min, t_max], ... }
    import json
    restart_list = json.loads(restart_info)
    # cast t_min to the index of nearest value in t_steps
    restart_list = {int(torch.argmin(abs(t_steps - v[2]), dim=0)): v for k, v in restart_list.items()}
    # dist.print0(f"restart configuration: {restart_list}")

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N_main -1

        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # Euler step.

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # ================= restart ================== #
        if i + 1 in restart_list.keys():
            restart_idx = i + 1

            for restart_iter in range(restart_list[restart_idx][1]):

                new_t_steps = get_steps(min_t=t_steps[restart_idx], max_t=restart_list[restart_idx][3],
                                        num_steps=restart_list[restart_idx][0], rho=rho)

                new_total_step = len(new_t_steps)
                if pfgmpp:
                    beta_gen = Beta(torch.FloatTensor([N / 2.]), torch.FloatTensor([net.D / 2.]))
                    sample_norm = beta_gen.sample(torch.Size([len(x_next)])).to(x_next.device).double()
                    # inverse beta distribution
                    inverse_beta = sample_norm / (1 - sample_norm)

                    sample_norm = torch.sqrt(inverse_beta) * t_steps[restart_idx] * np.sqrt(net.D)
                    gaussian = torch.randn(N).to(sample_norm.device)
                    unit_gaussian = gaussian / torch.norm(gaussian, p=2)
                    init_sample = unit_gaussian * sample_norm
                    x_next = x_next + init_sample.view_as(x_next) * S_noise
                else:
                    x_next = x_next + randn_like(x_next) * (new_t_steps[0] ** 2 - new_t_steps[-1] ** 2).sqrt() * S_noise


                for j, (t_cur, t_next) in enumerate(zip(new_t_steps[:-1], new_t_steps[1:])):  # 0, ..., N_restart -1

                    x_cur = x_next
                    gamma = restart_gamma if S_min <= t_cur <= S_max else 0
                    t_hat = net.round_sigma(t_cur + gamma * t_cur)

                    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
                    denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
                    d_cur = (x_hat - denoised) / (t_hat)
                    x_next = x_hat + (t_next - t_hat) * d_cur

                    # Apply 2nd order correction.
                    if j < new_total_step - 2 or new_t_steps[-1] != 0:
                        denoised = net(x_next, t_next, class_labels).to(torch.float64)
                        d_prime = (x_next - denoised) / t_next
                        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
        self.seeds = seeds
        self.device = device

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def rand_beta_prime(self, size, N=3072, D=128, **kwargs):
        # sample from beta_prime (N/2, D/2)
        # print(f"N:{N}, D:{D}")
        assert size[0] == len(self.seeds)
        latent_list = []
        beta_gen = Beta(torch.FloatTensor([N / 2.]), torch.FloatTensor([D / 2.]))
        for seed in self.seeds:
            torch.manual_seed(seed)
            sample_norm = beta_gen.sample().to(kwargs['device']).double()
            # inverse beta distribution
            inverse_beta = sample_norm / (1-sample_norm)

            if N < 256 * 256 * 3:
                sigma_max = 80
            else:
                raise NotImplementedError

            sample_norm = torch.sqrt(inverse_beta) * sigma_max * np.sqrt(D)
            gaussian = torch.randn(N).to(sample_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2)
            init_sample = unit_gaussian * sample_norm
            latent_list.append(init_sample.reshape((1, *size[1:])))

        latent = torch.cat(latent_list, dim=0)
        return latent

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--save_images',             help='only save a batch images for grid visualization',                     is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=0, show_default=True)
@click.option('--resume', 'resume',        help='resume ckpt', metavar='INT',                                       type=int, default=None, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
@click.option('--name',                    help='ckpt name',                                                        type=str, default=None, show_default=True)

# Restart parameters
@click.option('--restart_info', 'restart_info',               help='restart information', metavar='STR',            type =str, default='', show_default=True)
@click.option('--restart_gamma', 'restart_gamma',             help='restart gamma', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.05, show_default=True)

# PFGM related parameters
@click.option('--pfgmpp',          help='Train PFGM++', metavar='BOOL',                                             type=bool, default=False, show_default=True)
@click.option('--aug_dim',         help='additional dimension', metavar='INT',                                      type=click.IntRange(min=-1), default=128, show_default=True)

def main(outdir, seeds, class_idx, max_batch_size, save_images, pfgmpp, aug_dim, name, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    """


    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]


    stats = glob.glob(os.path.join(outdir, "*.pkl"))
    if stats == []:
        stats = glob.glob(os.path.join(outdir, "training-state-*.pt"))
    if stats == []:
        dist.print0(f'Please download the checkpoints to {outdir} first.')

    for ckpt_dir in stats:
            
        # Load network.
        dist.print0(f'Loading network from "{ckpt_dir}"...')
        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()

        if ckpt_dir[-3:] == 'pkl':
            # load pkl
            with dnnlib.util.open_url(ckpt_dir, verbose=(dist.get_rank() == 0)) as f:
                net = pickle.load(f)['ema'].to(device)
                ckpt_num = 000000
        else:
            # load pt
            data = torch.load(ckpt_dir, map_location=torch.device('cpu'))
            net = data['ema'].eval().to(device)
            ckpt_num = int(ckpt_dir[-9:-3])

        if pfgmpp:
            assert net.D == aug_dim

        if name is None:
            temp_dir = os.path.join(outdir, f'imgs')
        else:
            temp_dir = os.path.join(outdir, f'imgs_{name}')

        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

        # Loop over batches.
        dist.print0(f'Generating {len(seeds)} images to "{temp_dir}"...')
        for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            N = net.img_channels * net.img_resolution * net.img_resolution
            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_seeds)
            if pfgmpp:
                latents = rnd.rand_beta_prime([batch_size, net.img_channels, net.img_resolution, net.img_resolution],
                                    N=N,
                                    D=aug_dim,
                                    pfgmpp=pfgmpp,
                                    device=device)
            else:
                latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution],
                                    device=device)
            class_labels = None
            if net.label_dim:
                class_labels = torch.eye(net.label_dim, device=device)[
                    rnd.randint(net.label_dim, size=[batch_size], device=device)]
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1

            # Generate images.
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            with torch.no_grad():
                images = restart_sampler(net, latents, class_labels, randn_like=rnd.randn_like,
                                    pfgmpp=pfgmpp,  **sampler_kwargs)

            if save_images:
                # save a small batch of images
                images_ = (images + 1) / 2.
                print("len:", len(images))
                image_grid = make_grid(images_, nrow=int(np.sqrt(len(images))))
                save_image(image_grid, os.path.join(outdir, f'ode_images_{ckpt_num}.png'))
                exit(0)
                break
            # Save images.
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

            for seed, image_np in zip(batch_seeds, images_np):

                #image_dir = os.path.join(temp_dir, f'{seed - seed % 1000:06d}') if subdirs else outdir
                image_dir = os.path.join(temp_dir, f'{seed - seed % 1000:06d}')
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        # Done.
        torch.distributed.barrier()
        dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
