#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py \
#--steps 100 --scheduler DDPM --save_path /mnt/nfs_folder/imgs  --w 2
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py \
#--steps 200 --scheduler DDPM --save_path /mnt/nfs_folder/imgs  --w 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py \
--steps 30 --scheduler DDIM --save_path /mnt/nfs_folder/imgs  --w 1  --s_noise 1.01 --restart --name s_1.01

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py \
--steps 30 --scheduler DDIM --save_path /mnt/nfs_folder/imgs  --w 2  --s_noise 1.005 --restart --name s_1.005



#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py \
#--steps 30 --scheduler DDPM --save_path /mnt/nfs_folder/imgs --w 3.0
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py \
#--steps 51 --scheduler DDPM --save_path /mnt/nfs_folder/imgs --w 3.0
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py \
#--steps 70 --scheduler DDPM --save_path /mnt/nfs_folder/imgs --w 3.0
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py \
#--steps 200 --scheduler DDPM --save_path /mnt/nfs_folder/imgs --w 3.0