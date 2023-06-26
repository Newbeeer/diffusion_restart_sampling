
import os
from subprocess import PIPE, run
import argparse
import gdown


# setup arguments
parser = argparse.ArgumentParser(description='Restart Sampling')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'])
parser.add_argument('--method', type=str, default='vp', choices=['edm', 'vp', 'pfgmpp'])
args = parser.parse_args()


devices='0,1,2,3,4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = devices
batch_size = 192

def get_fid(num, name):

    if args.dataset == 'cifar10':
        fid_stats_path = './cifar10-32x32.npz'
    else:
        fid_stats_path = './imagenet-64x64.npz'

    for tries in range(2):
        fid_command = f'CUDA_VISIBLE_DEVICES=0 python fid.py  ./imgs/imgs_{name}  {fid_stats_path}  --num={num} > tmp_fid.txt'
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

    # If using PFGM++, setting up the augmentation dimension and the method flag
    pfgmpp = 1 if args.method == 'pfgmpp' else 0
    aug_dim = 2048 if args.method == 'pfgmpp' else -1
    s_noise = 1.003 if args.dataset == 'imagenet' else 1.0

    command = f"CUDA_VISIBLE_DEVICES={devices} torchrun --standalone --nproc_per_node={len(devices.split(','))} generate_restart.py --outdir=./imgs " \
              f"--restart_info='{kwargs['restart']}' --S_min=0.01 --S_max=1 --S_noise={s_noise} " \
              f"--steps={kwargs['steps']} --seeds=00000-{00000+num-1} --name={name} --batch={batch_size} --pfgmpp={pfgmpp} --aug_dim={aug_dim} #generate"
    print(command)
    os.system(command)
    if store:
        fid_score = get_fid(num, name)
        NFE = 0
        print("restart:", kwargs["restart"])
        dic = json.loads(kwargs["restart"])
        print("dic:", dic)
        for restartid in dic.keys():
            info = dic[restartid]
            NFE += 2 * info[1] * (info[0] - 1)
        NFE += (2 * kwargs['steps'] - 1)
        print(f'NFE:{NFE} FID_OF_{num}:{fid_score}')
        runs_dict[name] = {"fid": fid_score, "NFE": NFE, "Args": kwargs}


import random
import json

runs_dict = dict()
with open(f"params_{args.dataset}_{args.method}.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        sample_runs = 50000
        infos = line.split(' ')

        steps = int(infos[0])
        restart_info = ' '.join(infos[1:])
        print("restart_info:", restart_info)
        cur_name = random.randint(0, 20000000)
        generate(sample_runs, cur_name, True, restart=restart_info, steps=steps)

        with open(f"restart_runs_dict_{args.dataset}_{args.method}.json", "w") as f:
            json.dump(runs_dict, f)

