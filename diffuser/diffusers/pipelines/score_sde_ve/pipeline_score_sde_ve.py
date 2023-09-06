# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch

from ...models import UNet2DModel
from ...schedulers import ScoreSdeVeScheduler
from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class ScoreSdeVePipeline(DiffusionPipeline):
    r"""
    Parameters:
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image. scheduler ([`SchedulerMixin`]):
            The [`ScoreSdeVeScheduler`] scheduler to be used in combination with `unet` to denoise the encoded image.
    """
    unet: UNet2DModel
    scheduler: ScoreSdeVeScheduler

    def __init__(self, unet: UNet2DModel, scheduler: DiffusionPipeline):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        seed: int = 0,
        num_inference_steps: int = 200,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet

        # sigma_max = 1300
        # sigma_min = 0.002
        # rho = 7
        # step_indices = torch.arange(num_inference_steps, dtype=torch.float64).cuda()
        # t_steps = (sigma_max ** (1 / rho) + step_indices / (num_inference_steps - 1) * (
        #         sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        # t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float() # t_N = 0
        #

        # fix random seeds

        # measure the time
        import time
        start_time = time.time()


        def seed_everything(seed: int):
            import random, os
            import numpy as np

            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        seed_everything(seed)
        sample = torch.randn(shape) * self.scheduler.init_noise_sigma
        sample = sample.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        def euler(scores, x, sigma_t, sigma_t_next):
            # denoised = x + sigma_t ** 2 * scores
            # d_cur = (x - denoised) / sigma_t
            d_cur = - scores * sigma_t
            x_next = x + (sigma_t_next - sigma_t) * d_cur
            return x_next

        restart = True
        second = False
        sde = False
        restart_list = [i for i in range(50, 1000, 200)]
        restart_list_2 = [i for i in range(1, 50, 20)]
        restart_list = restart_list + restart_list_2
        restart_list += [0.1, 0.4]
        temp_list = []
        # map t_min to index
        for value in restart_list:
            temp_list.append(int(torch.argmin(abs(self.scheduler.sigmas - value), dim=0)))
        restart_list = temp_list
        print("restart_list:", restart_list)

        # print("total steps:", len(self.scheduler.timesteps))
        # print("total timesteps:", self.scheduler.timesteps)
        # print("total sigmas:", len(self.scheduler.sigmas))
        # total_steps = len(t_steps)

        nfe = 0
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
        #for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

            sample_cur = sample
            #print("step:", i, "t_cur:", t, "sigma_t:", self.scheduler.sigmas[i])
            sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=self.device)

            # # correction step
            if sde:
                for _ in range(self.scheduler.config.correct_steps):
                    model_output = self.unet(sample, sigma_t).sample
                    sample = self.scheduler.step_correct(model_output, sample, generator=generator).prev_sample
                    nfe += 1

            # prediction step
            #print("step:", i, "t_cur:", t_cur, "t_next:", t_next)

            model_output = model(sample, sigma_t).sample
            output = self.scheduler.step_pred(model_output, t, i, i+1, sample, generator=generator, sde=sde)
            nfe += 1
            sample = output.prev_sample
            sample_mean = output.prev_sample_mean

            if second and i < len(self.scheduler.timesteps) - 1:
                sigma_t_2 = self.scheduler.sigmas[i+1] * torch.ones(shape[0], device=self.device)
                model_output_2 = model(sample, sigma_t_2).sample
                output_2 = self.scheduler.step_pred(model_output, t, i, i+1, sample_cur, model_output_2=model_output_2, generator=generator)
                nfe += 1
                sample = output_2.prev_sample
                sample_mean = output_2.prev_sample_mean

            if restart and i in restart_list:

                if sigma_t < 5:
                    j = i - 15
                    K = 2
                    gap = 2
                elif sigma_t < 50:
                    j = i - 10
                    K = 2
                    gap = 2
                else:
                    j = i - 6
                    K = 1
                    gap = 1

                new_timesteps = self.scheduler.timesteps[j:i+1]
                new_sigma = self.scheduler.sigmas[j:]

                for restart_k in range(K):
                    #gap = 1
                    new_timesteps = torch.cat((new_timesteps[:-1][::gap], new_timesteps[-1].unsqueeze(0)))
                    new_sigma = torch.cat((new_sigma[:-1][::gap], new_sigma[-1].unsqueeze(0)))

                    sigma_i = self.scheduler.sigmas[i + 1]
                    sigma_j = self.scheduler.sigmas[j]
                    print(f'{sigma_i} -> {sigma_j}')
                    sample = sample + torch.randn_like(sample) * (sigma_j ** 2 - sigma_i ** 2).sqrt()

                    for k, t in enumerate(new_timesteps):
                        sample_cur = sample
                        sigma_t = new_sigma[k] * torch.ones(shape[0], device=self.device)

                        model_output = model(sample, sigma_t).sample
                        if k == len(new_timesteps) - 1:
                            output = self.scheduler.step_pred(model_output, t, j + k, j + k + 1, sample,
                                                              generator=generator, sde=sde)
                        else:
                            output = self.scheduler.step_pred(model_output, t, j + k, j + k + gap, sample,
                                                              generator=generator)

                        nfe += 1
                        sample = output.prev_sample
                        sample_mean = output.prev_sample_mean

                        if second:
                            sigma_t_2 = new_sigma[k + 1] * torch.ones(shape[0], device=self.device)
                            model_output_2 = model(sample, sigma_t_2).sample
                            interval = 1 if k == len(new_timesteps) - 1 else gap
                            output_2 = self.scheduler.step_pred(model_output, t, j + k, j + k + interval, sample_cur,
                                                                model_output_2=model_output_2, generator=generator)
                            nfe += 1
                            sample = output_2.prev_sample
                            sample_mean = output_2.prev_sample_mean


        # model_output = model(sample, t_cur).sample
            # d_cur = - model_output * t_cur
            # sample = sample_cur + 2 * (t_next - t_cur) * d_cur \
            #          + torch.sqrt(2 * t_cur) * torch.sqrt(t_cur - t_next) * randn_tensor(shape, generator=generator).to(sample.device)

            # if i < total_steps - 2:
            #     model_output = model(sample, t_next).sample
            #     d_prime = - model_output * t_next
            #     sample = sample_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        # measure elapsed time
        end_time = time.time()
        # measure the time in seconds
        elapsed_time = end_time - start_time
        print("elapsed time:", elapsed_time)

        print("nfe:", nfe)
        sample = sample_mean.clamp(0, 1)
        if output_type == 'tensor':
            return sample.cpu()

        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)
