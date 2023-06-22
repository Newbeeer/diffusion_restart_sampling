# evaluate FID

CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_endpoint=0.0.0.0:1202 --nproc_per_node=1 fid_edm.py calc \
--images /mnt/nfs_folder/imgs/ --ref /mnt/nfs_folder/fid-refs/coco.npz --num 5000


CUDA_VISIBLE_DEVICES=0 python fid.py   /mnt/nfs_folder/imgs/ /mnt/nfs_folder/fid-refs/coco.npz


# evaluate clip score / aesthetic score

python eval_clip_score.py --csv_path /mnt/nfs_folder/imgs/ --dir_path /mnt/nfs_folder/imgs