### Standard Benchmarks (CIFAR-10, ImageNet-64)

**The working directory for standard benchmarks is under `./benchmarks`**

#### 1. Preparing datasets and checkpoints

**CIFAR-10:** Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```

**ImageNet:** Download the [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \
    --dest=datasets/imagenet-64x64.zip --resolution=64x64 --transform=center-crop
python fid.py ref --data=datasets/imagenet-64x64.zip --dest=fid-refs/imagenet-64x64.npz
```

Alternatively, you could consider downloading the FID statistics at [CIFAR-10-FID](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz) and [ImageNet-$64\times 64$-FID](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz).



Please download the CIFAR-10 or ImageNet checkpoints from [EDM](https://github.com/NVlabs/edm) repo or [PFGM++](https://github.com/Newbeeer/pfgmpp) repo. For example,

| Dataset                | Method                          | Path                                                         |
| ---------------------- | ------------------------------- | ------------------------------------------------------------ |
| CIFAR-10               | VP unconditional                | [path](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl) |
| CIFAR-10               | PFGM++ ($D=2048$) unconditional | [path](https://drive.google.com/drive/folders/1sZ7vh7o8kuXfFjK8ROWXxtEZi8Srewgo) |
| ImageNet $64\times 64$ | EDM conditional                 | [path](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl) |



#### 2. Generate

Generating a large number of images can be time-consuming; the workload can be distributed across multiple GPUs by launching the above command using `torchrun`. Before generation, please make sure the checkpoint is downloaded in the `./benchmarks/imgs` folder

```shell
torchrun --standalone --nproc_per_node=8 generate_restart.py --outdir=./imgs \ 
--restart_info='{restart_config}' --S_min=0.01 --S_max=1 --S_noise 1.003 \
--steps={steps} --seeds=00000-49999 --name={name} (--pfgmpp=1) (--aug_dim={D})


restart_config: configuration for Restart (details below)
steps: number of steps in the main backward process, default=18
name: name of experiments (for FID evaluation)
pfgmpp: flag for using PFGM++
D: augmented dimension in PFGM++
```

The `restart_info` is in the format of  $\lbrace i: [N_{\textrm{Restart},i}, K_i, t_{\textrm{min}, i}, t_{\textrm{max}, i}] \rbrace_{i=0}^{l-1}$ , such as `{"0": [3, 2, 0.06, 0.30]}`. Please refer to Table 3 (CIFAR-10) and Table5 (ImageNet-64) for detailed configuration. For example, on uncond. EDM cond. ImageNet-64, with NFE=203, FID=1.41, the command line is:

```shell
torchrun --standalone --nproc_per_node=8 generate_restart.py --outdir=./imgs \ 
--restart_info='{"0": [4, 1, 19.35, 40.79], "1": [4, 1, 1.09, 1.92], "2": [4, 5, 0.59, 1.09], "3": [4, 5, 0.30, 0.59], "4": [6, 6, 0.06, 0.30]}' --S_min=0.01 --S_max=1 --S_noise 1.003 \
--steps=36 --seeds=00000-49999 --name=imagenet_edm
```

We also provide the extentive Restart configurations in `params_cifar10_vp.txt`, `params_imagenet_edm.txt`, corresponding to Table 3 (CIFAR-10) and Table5 (ImageNet-64) respectively. Each line in these `txt` is in the form of $N_{\textrm{main}} \quad \lbrace i: [N_{\textrm{Restart},i}, K_i, t_{\textrm{min}, i}, t_{\textrm{max}, i}]\rbrace_{i=0}^{l-1}$. To sweep the Restart configurations in the `txt` files, please run

```shell
python3 hyperparams.py --dataset {dataset} --method {method}

dataset: cifar10 | imagenet
method: edm | vp | pfgmpp
```

The above sweeping will reproduce the results in the following figure (Fig 3 in the paper):

![schematic](../assets/fig_3.png)

#### 3. Evaluation

For FID evaluation, please run:

```shell
python fid.py  ./imgs/imgs_{name} stats_path

name: name of experiments (specified in geneation command line)
stats_path: path to FID statistics, such as ./cifar10-32x32.npz or ./imagenet-64x64.npz
```



