#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
#  --steps=18  --seeds=0-49999 --use_pickle=1 --name=sde_18 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
#  --steps=32  --seeds=0-49999 --use_pickle=1 --name=sde_32 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=18  --seeds=50000-99999 --use_pickle=1 --name=sde_18_2 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=32  --seeds=50000-99999 --use_pickle=1 --name=sde_32_2 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=64  --seeds=50000-99999 --use_pickle=1 --name=sde_64_2 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=128  --seeds=50000-99999 --use_pickle=1 --name=sde_128_2 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=256  --seeds=50000-99999 --use_pickle=1 --name=sde_256_2 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=512  --seeds=50000-99999 --use_pickle=1 --name=sde_512_2 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=18  --seeds=100000-149999 --use_pickle=1 --name=sde_18_3 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=32  --seeds=100000-149999 --use_pickle=1 --name=sde_32_3 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=64  --seeds=100000-149999 --use_pickle=1 --name=sde_64_3 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=128  --seeds=100000-149999 --use_pickle=1 --name=sde_128_3 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=256  --seeds=100000-149999 --use_pickle=1 --name=sde_256_3 --batch 256 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_xyl.py --outdir=./imgs \
  --steps=512  --seeds=100000-149999 --use_pickle=1 --name=sde_512_3 --batch 25 --S_churn 80 --S_min 0.05 --S_max 50 --S_noise 1.003
