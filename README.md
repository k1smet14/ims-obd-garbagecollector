# 안현진

### [code]SoftEnsemble.ipynb
(4.30) 앙상블용 코드 사용하기 쉽게 작성

### [code]DL3P+resnext50_resize_iouCE.ipynb
baseline 실험 코드

### [code]DL3P+resnext50_resize_iouCE_swsl_classMix.ipynb
(4.29) classmix 구현 코드 : MixDataset사용, classMix함수 구현, train부분 코드 약간 수정

### train_wandb.py
wandb autoML 실행 코드 (1차 수정)

### dataloader.py
CustomDataset <br>
(4.29) MixDataset 추가

### scheduler.py
custom scheduler : CosineAnnealingWarmUpRestarts() <br>
출처: https://gaussian37.github.io/dl-pytorch-lr_scheduler/

### evaluate.py
metric 및 validation 함수 <br>
배치 기준 -> 통합 mIoU 추가 (validation2) <br>
(4.29) validation2 : 배치기준, 전체기준 mIoU 모두 계산
(4.30) validation3 : 배치기준, 전체기준, 이미지기준 3가지 모두 mIoU 

### utils.py
save, load, submit, calculate_parameter

### loss.py
(4.29) Focal, iou_CE, Dice_CE
