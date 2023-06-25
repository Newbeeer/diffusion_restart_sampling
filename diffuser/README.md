### Stable Diffusion

TODO: merge into the diffuser repo.

**The working directory for standard benchmarks is under `./diffuser`**

![schematic](../assets/fig_5.png)

#### 1. Data processing:

- Step 1: Follow the instruction at the head of `data_process.py`
- Step 2: Run `python3 data_process.py` to randomly sampled 5K image-text pair from COCO validation set.

- Step 3: Calculate FID statistics `python fid_edm.py ref --data=path-to-coco-subset --dest=./coco.npz`

#### 2. Generate

```
torchrun --rdzv_endpoint=0.0.0.0:1201 --nproc_per_node=8 generate.py 
	--steps: number of sampling steps (default=50) 
	--scheduler: baseline method (DDIM | DDPM | Heun)
	--save_path: path to save images
	--w: classifier-guidance weight ({2,3,5,8})
	--name: name of experiments
	--restart
```



#### 3. Evaluation

- FID score

  ```sh
  python3 fid.py {path} ./coco.npz
  
  path: path to the directory of generated image
  ```

- Aesthetic score & CLIP score

  ```shell
  python3 eval_clip_score.py --csv_path {path}/subset.csv --dir_path {path}
  
  path: path to the directory of generated image
  ```

