CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=18  --seeds=50000-99999 --use_pickle=1 --name=ode_18_2 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=32  --seeds=50000-99999 --use_pickle=1 --name=ode_32_2 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=64  --seeds=50000-99999 --use_pickle=1 --name=ode_64_2 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=128  --seeds=50000-99999 --use_pickle=1 --name=ode_128_2 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=256  --seeds=50000-99999 --use_pickle=1 --name=ode_256_2 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=512  --seeds=50000-99999 --use_pickle=1 --name=ode_512_2 --batch 256


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=18  --seeds=100000-149999 --use_pickle=1 --name=ode_18_3 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=32  --seeds=100000-149999 --use_pickle=1 --name=ode_32_3 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=64  --seeds=100000-149999 --use_pickle=1 --name=ode_64_3 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=128  --seeds=100000-149999 --use_pickle=1 --name=ode_128_3 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=256  --seeds=100000-149999 --use_pickle=1 --name=ode_256_3 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs_cifar \
  --steps=512  --seeds=100000-149999 --use_pickle=1 --name=ode_512_3 --batch 25