# 안현진
### (05.04) K-Fold용 pseudo 데이터 업로드
pseudo_kfold_all.npy, pseudo_kfold_anns.npy 파일을 다운받습니다. <br>
1. input/data/ 경로에 두 파일을 위치해주세요.
2. [code]KFold+Pseudo_example.ipynb 코드를 다운받아 사용하시면 됩니다.
3. train 함수에 model save 코드가 있는 점 주의해서 수정해주세요.


### (05.03) Pseudo data 업로드
test데이터를 segmentation하여 만든 mask파일 760장이 들어있습니다.<br>
1. pseudo_masks.zip 파일을 다운받은 후 압축을 푸셔서, input/data/pseudo_masks 경로로 만들어주세요.
2. pseudo_imgs_path.npy 파일을 다운받으신 후 input/data/pseudo_imgs_path.npy로 위치해주세요.
3. pseudo_mask 사용 예제.ipynb 코드를 보시면 PseudoTrainset 클래스가 있습니다. 복사해서 기존의 CustomDataset처럼 사용하시면 됩니다.

### (05.03) [code]KFold_objectNumSplit.ipynb
이미지당 개체수를 기준으로 5Fold split<br>
이미지가 5장 미만인 개체 수는 100으로 통합<br>


### [code]SoftEnsemble.ipynb
(4.30) 앙상블용 코드 사용하기 쉽게 작성<br>
(5.01) scaling, TTA 앙상블 추가 (dataloader.py 업데이트 필요)<br>
(5.04) SoftEnsemble_4TTA.ipynb 추가. (normal, flip, clockRotate, counterClockRotate)

### [code]DL3P+resnext50_resize_iouCE.ipynb
baseline. 실험용 코드

### [code]DL3P+resnext50_resize_iouCE_swsl_classMix.ipynb
(4.29) classmix 구현 코드 : MixDataset사용, classMix함수 구현, train부분 코드 약간 수정 <br>
(5.01) weighted classMix를 위해 classMix함수 수정

### train_wandb.py
wandb autoML 실행 코드 (1차 수정)

### dataloader.py
CustomDataset <br>
(4.29) MixDataset 추가<br>
(5.01) 앙상블을 위한 EnsembleDataset 추가<br>
       weighted classMix를 위해 변수 및 함수 하나씩 추가 및 MixDataset 수정

### scheduler.py
custom scheduler : CosineAnnealingWarmUpRestarts() <br>
출처: https://gaussian37.github.io/dl-pytorch-lr_scheduler/

### evaluate.py
metric 및 validation 함수 <br>
배치 기준 -> 통합 mIoU 추가 (validation2) <br>
(4.29) validation2 : 배치기준, 전체기준 mIoU 모두 계산<br>
(4.30) validation3 : 배치기준, 전체기준, 이미지기준 3가지 모두 mIoU 

### utils.py
save, load, submit, calculate_parameter

### loss.py
(4.29) Focal, iou_CE, Dice_CE
