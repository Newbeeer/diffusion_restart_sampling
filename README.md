# Restart sampling

Restart Sampling for Improving Generative Processes

by Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, Tommi S. Jaakkola



Generative processes that involve solving differential equations, such as diffusion models, frequently necessitate balancing speed and quality. ODE-based samplers are fast but plateau in performance while SDE-based samplers deliver higher sample quality at the cost of increased sampling time.  We attribute this difference to sampling errors: ODE-samplers involve smaller discretization errors while stochasticity in SDE contracts accumulated errors. Based on these findings, **we propose a novel sampling algorithm called *Restart* in order to better balance discretization errors and contraction.** The sampling method alternates between adding substantial noise in additional forward steps and strictly following a backward ODE.

Empirically, **Restart sampler surpasses previous diffusion SDE and ODE samplers in both speed and accuracy**. Restart not only outperforms the previous best SDE results, but also accelerates the sampling speed by 10-fold / 2-fold on CIFAR-10 / ImageNet $64{\times} 64$. In addition, it attains significantly better sample quality than ODE samplers within comparable sampling times. Moreover, **Restart better balances text-image alignment/visual quality versus diversity** than previous samplers in the large-scale text-to-image **Stable Diffusion** model pre-trained on LAION $512{\times} 512$.



![schematic](assets/restart.png)



code coming soon...





