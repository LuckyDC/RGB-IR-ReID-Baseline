# A Strong Baseline for RGD-Infrared Cross-Modality Person Re-Identification

## Dependency
* Python 3.7
* PyTorch 1.2
* Ignite 
* Apex
* Yacs

## Utilization
Run
```shell script
CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg configs/softmax.yml
```
to train the model with SoftMax classification only.

Or run
```shell script
CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg configs/softmax-triplet.yml
```
to train the model with classification and triplet loss.

## Performance

We evaluate the performance on [SYSU-MM01](https://github.com/wuancong/SYSU-MM01) under the setting of  **one-shot** & **all-search**.

| model             | mAP | rank-1 | rank-5 | rank-10 | rank-20 |
| ----------------- | ------ | ------ | ------- | ------- | ------- |
| softmax only      | 39.90 | 38.90 | 69.08 | 80.34 | 89.45 |
| softmax + triplet | 41.85 | 41.55 | 72.65 | 84.27 | 92.78 |
| softmax only + RE      | 45.80 | 46.57 | 77.18 | 86.79 | 93.92 |
| softmax + triplet + RE | 45.40 | 46.20 | 75.80 | 86.17 | 93.77 |

RE denotes RandomErasing augmentation.
We adopt one-stream network and find that totally two-stream network leads to inferior performance.
Furthermore, we find that just making lower layers (e.g up to layer2) independent can lead to similar performance as one-stream network. 

## Reference 

[L1aoXingyu/reid_baseline](https://github.com/L1aoXingyu/reid_baseline)
