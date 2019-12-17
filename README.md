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
| softmax only      | 0.399001 | 0.388982 | 0.690849 | 0.803392 | 0.894452 |
| softmax + triplet | 0.418460 | 0.415488 | 0.726479 | 0.842677 | 0.92782 |
| softmax only + RE      | 0.458011 | 0.465685 | 0.771838 | 0.867946 | 0.939232 |
| softmax + triplet + RE | 0.454035 | 0.461977 | 0.757954 | 0.861662 | 0.93770 |

RE denotes RandomErasing augmentation.

## Reference 

[L1aoXingyu/reid_baseline](https://github.com/L1aoXingyu/reid_baseline)