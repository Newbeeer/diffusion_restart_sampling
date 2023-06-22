CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=18  --seeds=00000-49999 --use_pickle=1 --name=restart_40 --batch 256 --restart 40 --S_noise 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=18  --seeds=00000-49999 --use_pickle=1 --name=restart_60 --batch 256 --restart 60  --S_noise 1

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
#  --steps=64  --seeds=00000-49999 --use_pickle=1 --name=ode_64_2 --batch 256
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
#  --steps=128  --seeds=00000-49999 --use_pickle=1 --name=ode_128_2 --batch 256
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
#  --steps=256  --seeds=00000-49999 --use_pickle=1 --name=ode_256_2 --batch 256
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
#  --steps=512  --seeds=00000-49999 --use_pickle=1 --name=ode_512_2 --batch 256

