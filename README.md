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

| model             | rank-1 | rank-5 | rank-10 | rank-20 |
| ----------------- | ------ | ------ | ------- | ------- |
| softmax only      |        |        |         |         |
| softmax + triplet |        |        |         |         |
|                   |        |        |         |         |
|                   |        |        |         |         |

