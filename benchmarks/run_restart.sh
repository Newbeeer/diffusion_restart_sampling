#!/bin/bash
for i in $(seq 5 7);
do
  torchrun --rdzv_endpoint=0.0.0.0:1203 --nproc_per_node=8 generate_dpm_solver.py --outdir=./imgs --restart_info='{"0": [3, 1, 0.06, 1]}' --S_min=0.01 --S_max=1 --S_noise 1.003 --steps=$i --seeds=50000-99999 --name step_3_1_1_dpm3_3_$i
done