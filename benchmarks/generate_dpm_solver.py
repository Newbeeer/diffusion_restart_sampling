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
    pfgmpp=False, restart_info="", restart_gamma=0, schedule='vp',
):

    def get_steps(min_t, max_t, num_steps, rho):

         step_indices = torch.arange(num_steps, dtype=torch.float, device=latents.device)
         t_steps = (max_t ** (1 / rho) + step_indices / (num_steps - 1) * (min_t ** (1 / rho) - max_t ** (1 / rho))) ** rho
         return t_steps

    N = net.img_channels * net.img_resolution * net.img_resolution
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)


    if schedule == 'vp':

        epsilon_s = 1e-3

        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        sigma_min = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_max = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)

        # Compute corresponding betas for VP.
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()

        alpha = lambda t: s(t)
        sigma_hat = lambda t: sigma(t) * alpha(t)

        sigma2t = lambda s: (- vp_beta_min + (vp_beta_min ** 2 + 2 * vp_beta_d * torch.log(1 + s**2)) ** 0.5) / vp_beta_d
        t2lambda = lambda t: torch.log(alpha(t)/sigma_hat(t))
        lambda2t = lambda l: (- vp_beta_min + (vp_beta_min ** 2 + 2 * vp_beta_d * torch.log(1/torch.exp(2*l)+1)) ** 0.5) / vp_beta_d




    def sigma_to_lambda(sigma):
        return t2lambda(sigma2t(sigma))

    def lambda_to_sigma(l):
        return sigma(lambda2t(l))


    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    #print(t_steps)
    # the following schedule in DPM-solver paper performs worse than the EDM schedule
    # if schedule == 'vp':
    #     l_upper = t2lambda(torch.tensor([1.]))
    #     l_lower = t2lambda(torch.tensor([1e-3]))
    #     logSNR_steps = torch.linspace(float(l_upper.numpy()), float(l_lower.numpy()), num_steps + 1).to(t_steps.device)
    #     t_steps = lambda_to_sigma(logSNR_steps)

    total_step = len(t_steps)

    def dpm_solver_1(net, x_hat, t_next, t_hat):

        l_next = sigma_to_lambda(t_next)
        l_hat = sigma_to_lambda(t_hat)
        h = l_next - l_hat

        t_next_ = sigma2t(t_next)
        t_hat_ = sigma2t(t_hat)

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        # print(sigma_hat(t_next_)/alpha(t_next_) * (torch.exp(h) - 1), t_hat - t_next, t_hat)
        x_next = x_hat - sigma_hat(t_next_)/alpha(t_next_) * (torch.exp(h) - 1) * d_cur

        return x_next

    def dpm_solver_2(net, x_hat, t_next, t_hat, r_1=0.5):

        l_next = sigma_to_lambda(t_next)
        l_hat = sigma_to_lambda(t_hat)
        h = l_next - l_hat

        t_inter_ = lambda2t((l_hat + r_1 * h))
        t_inter = sigma(t_inter_)
        t_next_ = sigma2t(t_next)
        t_hat_ = sigma2t(t_hat)

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_inter = x_hat - sigma_hat(t_inter_)/alpha(t_inter_) * (torch.exp(r_1 * h) - 1) * d_cur

        denoised = net(x_inter, t_inter, class_labels).to(torch.float64)
        d_inter = (x_inter - denoised) / t_inter
        x_next = x_hat - (1-1/(2*r_1)) * sigma_hat(t_next_)/alpha(t_next_) * (torch.exp(h) - 1) * d_cur - 1/(2*r_1) * sigma_hat(t_next_)/alpha(t_next_) * (torch.exp(h) - 1) * d_inter

        return x_next

    def dpm_solver_3(net, x_hat, t_next, t_hat, r_1=1./3, r_2=2./3):

        l_next = sigma_to_lambda(t_next)
        l_hat = sigma_to_lambda(t_hat)
        h = l_next - l_hat

        t_inter_ = lambda2t((l_hat + r_1 * h))
        t_inter = sigma(t_inter_)

        t_inter_2_ = lambda2t((l_hat + r_2 * h))
        t_inter_2 = sigma(t_inter_2_)

        t_next_ = sigma2t(t_next)
        t_hat_ = sigma2t(t_hat)

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_inter = x_hat - sigma_hat(t_inter_)/alpha(t_inter_) * (torch.exp(r_1 * h) - 1) * d_cur

        denoised = net(x_inter, t_inter, class_labels).to(torch.float64)
        d_inter = (x_inter - denoised) / t_inter

        D_1 = d_inter - d_cur

        x_inter_2 = x_hat - sigma_hat(t_inter_2_)/alpha(t_inter_2_) * (torch.exp(r_2 * h) - 1) * d_cur \
                    - r_2/r_1 * sigma_hat(t_inter_2_)/alpha(t_inter_2_) * ((torch.exp(r_2 * h)-1)/(r_2 * h) - 1) * D_1

        denoised = net(x_inter_2, t_inter_2, class_labels).to(torch.float64)
        d_inter_2 = (x_inter_2 - denoised) / t_inter_2

        D_2 = d_inter_2 - d_cur

        x_next = x_hat - sigma_hat(t_next_)/alpha(t_next_) * (torch.exp(h) - 1) * d_cur \
                 - 1/(r_2) * sigma_hat(t_next_)/alpha(t_next_) * ((torch.exp(h)-1)/h - 1) * D_2

        return x_next


    if pfgmpp:
        x_next = latents.to(torch.float64)
    else:
        x_next = latents.to(torch.float64) * t_steps[0]
        # Main sampling loop.

    # {[num_steps, number of restart iteration (K), t_min, t_max], ... }
    import json

    restart_list = json.loads(restart_info) if restart_info != '' else {}
    # cast t_min to the index of nearest value in t_steps
    restart_list = {int(torch.argmin(abs(t_steps - v[2]), dim=0)): v for k, v in restart_list.items()}
    #dist.print0(f"restart configuration: {restart_list}")
    nfe = 0
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N_main -1
        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # Euler step.

        if i < total_step - 3:
            x_next = dpm_solver_3(net, x_hat, t_next, t_hat)
            nfe += 3
        elif i < total_step - 2:
            x_next = dpm_solver_2(net, x_hat, t_next, t_hat)
            nfe += 2
        else:
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            nfe += 1

        # # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     denoised = net(x_next, t_next, class_labels).to(torch.float64)
        #     d_prime = (x_next - denoised) / t_next
        #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # ================= restart ================== #
        if i + 1 in restart_list.keys():
            restart_idx = i + 1

            for restart_iter in range(restart_list[restart_idx][1]):

                new_t_steps = get_steps(min_t=t_steps[restart_idx], max_t=restart_list[restart_idx][3],
                                        num_steps=restart_list[restart_idx][0], rho=rho)
                dist.print0(f"restart at {restart_idx} with {new_t_steps}")
                new_total_step = len(new_t_steps)
                if pfgmpp:
                    # convert sigma to radius by r=sigma*sqrt(D)
                    old_r = new_t_steps[-1] * np.sqrt(net.D)
                    new_r = new_t_steps[0] * np.sqrt(net.D)

                    z = torch.randn(len(x_next), net.D).to(x_next.device)
                    z /= z.norm(p=2, dim=1, keepdim=True)
                    z *= old_r

                    # uniform sampling on the sphere in N+D dimension
                    dir = torch.randn(len(x_next), N+net.D).to(x_next.device)
                    dir /= dir[:, N:].norm(dim=1, keepdim=True)

                    # determine the length of the move
                    dot = (z * dir[:, N:]).sum(dim=1)
                    randint = torch.randint(1, size=(x_next.shape[0],)).float().to(x_next.device)
                    moves = (-dot + torch.sqrt(dot ** 2 + (new_r ** 2 - old_r ** 2))) * randint + (
                                -dot - torch.sqrt(dot ** 2 + (new_r ** 2 - old_r ** 2))) * (1 - randint)
                    moves = moves.view(len(x_next), 1, 1, 1).to(x_next.device)

                    # apply PFGM++ perturbation kernel from old radius to new radius
                    x_next = x_next + moves * dir[:, :N].view_as(x_next) * S_noise
                else:
                    x_next = x_next + randn_like(x_next) * (new_t_steps[0] ** 2 - new_t_steps[-1] ** 2).sqrt() * S_noise


                for j, (t_cur, t_next) in enumerate(zip(new_t_steps[:-1], new_t_steps[1:])):  # 0, ..., N_restart -1

                    x_cur = x_next
                    gamma = restart_gamma if S_min <= t_cur <= S_max else 0
                    t_hat = net.round_sigma(t_cur + gamma * t_cur)
                    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

                    x_next = dpm_solver_3(net, x_hat, t_next, t_hat)
                    nfe += 3

    print(f"nfe: {nfe}")
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
