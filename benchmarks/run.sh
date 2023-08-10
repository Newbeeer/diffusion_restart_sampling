#!/bin/bash
for i in $(seq 5 13);
do
  torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate_dpm_solver.py --outdir=./imgs --restart_info='' --S_min=0.01 --S_max=1 --S_noise 1.003 --steps=$i --seeds=50000-99999 --name step_dpm3_$i
done