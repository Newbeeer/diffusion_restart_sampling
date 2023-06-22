CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_sde.py --outdir=./imgs_cifar \
  --steps=32  --seeds=0-49999 --use_pickle=1 --name=vanilla_32 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_sde.py --outdir=./imgs_cifar \
  --steps=64  --seeds=0-49999 --use_pickle=1 --name=vanilla_64 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_sde.py --outdir=./imgs_cifar \
  --steps=128  --seeds=0-49999 --use_pickle=1 --name=vanilla_128 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_sde.py --outdir=./imgs_cifar \
  --steps=256  --seeds=0-49999 --use_pickle=1 --name=vanilla_256 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_sde.py --outdir=./imgs_cifar \
  --steps=512  --seeds=0-49999 --use_pickle=1 --name=vanilla_512 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_sde.py --outdir=./imgs_cifar \
  --steps=1024  --seeds=0-49999 --use_pickle=1 --name=vanilla_1024 --batch 256