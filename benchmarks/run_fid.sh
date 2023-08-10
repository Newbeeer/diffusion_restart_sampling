#!/bin/bash
for i in $(seq 8 11);
do
  python fid.py  ./imgs/imgs_step_3_1_0.3_dpm3_3_$i cifar10-32x32.npz
done