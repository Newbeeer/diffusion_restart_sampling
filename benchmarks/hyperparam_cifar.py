
import os
from subprocess import PIPE, run
import argparse
parser = argparse.ArgumentParser(description='SDE ImageNet')
parser.add_argument('--instance', type=int, default=5)
args = parser.parse_args()

# set CUDA_VISIBLE_DEVICES=0,1,6
devices='0,1,2,3,4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = devices
# sample_runs = 1024 * 4 # 1024
batch_size = 192 # 768 #(2048 + len(devices.split(',')) - 1) // len(devices.split(','))
instance_id = args.instance

def get_fid(num, name):
    for tries in range(2):
        #fid_command = f"CUDA_VISIBLE_DEVICES={devices} torchrun --standalone --nproc_per_node={len(devices.split(','))} fid.py calc --num={num} --images=./imgs_cifar/ckpt_000000_{name} --name=./imgs_cifar/ckpt_000000_{name} --ref=./imagenet-64x64.npz > tmp_fid.txt"
        # result = run(fid_command, shell=True)
        fid_command = f'CUDA_VISIBLE_DEVICES=0 python fid.py  ./imgs_cifar/ckpt_000000_{name}  ./cifar10-32x32.npz  --num={num} > tmp_fid.txt'
        print('----------------------------')
        print(fid_command)
        os.system(fid_command)
        with open("tmp_fid.txt", "r") as f:
            output = f.read()
            print(output)
        try:
            fid_score = float(output.split()[-1])
            return fid_score
        except:
            print("FID computation failed, trying again")
        print('----------------------------')
    return 1e9


runs_dict = dict()

def generate(num, name, store = True, **kwargs):
    # print("running", kwargs)
    # name = get_name(**kwargs)
    command = f"CUDA_VISIBLE_DEVICES={devices} torchrun --standalone --nproc_per_node={len(devices.split(','))} generate_restart.py --outdir=./imgs_cifar --restart_info='{kwargs['restart']}' --S_min=0.01 --S_max=1 --S_noise=1.0 --S_churn={kwargs['churn']} --steps={kwargs['steps']} --restart_gamma={kwargs['restart_gamma']} --seeds=50000-{50000+num-1} --use_pickle=1 --name={name} --batch={batch_size} #generate"
    print(command)
    os.system(command)
    # {4: [1, 40.7864, 3],  10: [1, 1.92, 4], 11: [5, 1.088, 4], 12: [5, 0.5853, 4], 14: [10, 0.2964, 4]}
    if store:
        fid_score = get_fid(num, name)
        NFE = 0
        print("restart:", kwargs["restart"])
        dic = json.loads(kwargs["restart"])
        print("dic:", dic)
        for restartid in dic.keys():
            info = dic[restartid]
            NFE += 2 * info[0] * (info[2] - 1)
        NFE += (2 * kwargs['steps'] - 1)
        print(f'NFE:{NFE} FID_OF_{num}:{fid_score}')
        runs_dict[name] = {"fid": fid_score, "NFE": NFE, "Args": kwargs}


import random
import math
import json
import torch
import copy

runs_dict = dict()
with open(f"try_params_{instance_id}.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        sample_runs = 50000
        infos = line.split(' ')
        churn = infos[-1]
        churn = churn.replace('\n', '')
        steps = int(infos[-3])
        restart_gamma = float(infos[-2])
        restart_info = ' '.join(infos[:-3])
        cur_name = random.randint(0, 20000000)
        generate(sample_runs, cur_name, True, restart=restart_info, churn=churn, steps=steps, restart_gamma=restart_gamma)

        with open(f"restart_runs_dict_{instance_id}.json", "w") as f:
            json.dump(runs_dict, f)
