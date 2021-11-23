# A Strong Baseline for RGD-Infrared Cross-Modality Person Re-Identification

## Dependency
* Python 3.7
* PyTorch 1.10
* Ignite 0.4.7
* Yacs

## Utilization
Download [SYSU-MM01](https://github.com/wuancong/SYSU-MM01) dataset and uncompress it.
Change the entry `data_root` in configs/default.py to the path of the dataset.
Put the [rand_perm_cam.mat](https://github.com/wuancong/SYSU-MM01/blob/master/evaluation/data_split/rand_perm_cam.mat) in `exp` directory in dataset root. This file is used to assign gallery items for each trial while testing.
Run
```shell script
CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg configs/baseline.yml
```

## Performance

We evaluate the performance on [SYSU-MM01](https://github.com/wuancong/SYSU-MM01) under the setting of  **one-shot** & **all-search**.

| model             | mAP | rank-1 | rank-5 | rank-10 | rank-20 |
| ----------------- | ------ | ------ | ------- | ------- | ------- |
| baseline | 54.60	| 57.51	| 82.77 |	90.05 | 95.28 |


## Reference 

[L1aoXingyu/reid_baseline](https://github.com/L1aoXingyu/reid_baseline)
