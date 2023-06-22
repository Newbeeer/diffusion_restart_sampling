CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_nvanilla=8 generate_sde.py --outdir=./imgs \
  --steps=32  --seeds=10000-59999 --use_pickle=1 --name=vanilla_32 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_nvanilla=8 generate_sde.py --outdir=./imgs \
  --steps=64  --seeds=10000-59999 --use_pickle=1 --name=vanilla_64 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_nvanilla=8 generate_sde.py --outdir=./imgs \
  --steps=128  --seeds=10000-59999 --use_pickle=1 --name=vanilla_128 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_nvanilla=8 generate_sde.py --outdir=./imgs \
  --steps=256  --seeds=10000-59999 --use_pickle=1 --name=vanilla_256 --batch 256

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_nvanilla=8 generate_sde.py --outdir=./imgs \
  --steps=512  --seeds=10000-59999 --use_pickle=1 --name=vanilla_512 --batch 256

