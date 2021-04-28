# 안현진

### [code]DL3P+resnext50_resize_iouCE.ipynb
baseline 실험 코드

### train_wandb.py
wandb autoML 실행 코드 (1차 수정)

### dataloader.py
CustomDataset, get_class_weight, get_weighted_sampler

### scheduler.py
custom scheduler : CosineAnnealingWarmUpRestarts() <br>
출처: https://gaussian37.github.io/dl-pytorch-lr_scheduler/

### evaluate.py
metric 및 validation 함수 <br>
배치 기준 -> 통합 mIoU 추가 (validation2)

### utils.py
save, load, submit, calculate_parameter
