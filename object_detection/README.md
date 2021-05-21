# Content
- [How to use mosaic augmentation with mmdetection](https://github.com/bcaitech1/p3-ims-obd-garbagecollector/tree/MinKi/object_detection#how-to-use-mosaic)
- [How to use swin transformer](https://github.com/bcaitech1/p3-ims-obd-garbagecollector/tree/MinKi/object_detection#how-to-use-swin_transformer)
- [How to make mix data](https://github.com/bcaitech1/p3-ims-obd-garbagecollector/tree/MinKi/object_detection#how-to-make-mix-data)

<br>
<br>

# How to use swin_transformer
1. Step One - install swin_transformer
```bash
cd swin_object_detection
python setup.py develop
```
2. Step Two - download pretrained model
```bash
cd swin_object_detection
mkdir pretrained
cd pretrained
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth

# please read original repo 
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
```
3. Step Three - train with swin_object_detection/train.ipynb

<br>
<br>

# How to use Mosaic 
- cfg 파일을 따라가다 아래와 같이 import를 하는데 `'../_base_/datasets/coco_detection.py'` 부분을 `'../_base_/datasets/coco_detection_mosaic.py'`로 변경해주시면 됩니다.

```python
# configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco_mosaic.py

# 기존 _base_
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# 변경 후 _base_
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_mosaic.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```
- 혹은 `configs/trash/dataset.py`내용을 아래와 같이 변경하셔도 됩니다. 저같은 경우는 `dataset_mosaic.py`를 새로 만들었습니다.
```python
# 변경전
dataset_type = 'CocoDataset'
data_root = '../../input/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test_private.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

# 변경후
dataset_type = 'MosaicCocoDataset'
data_root = '/opt/ml/input/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImageFromFile'),
    dict(type='LoadMultiAnnotations', with_bbox=True),
    dict(type='Mosaic'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test_private.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
```
- train_pipleline이 아래와 같이 변경되었기 때문에 기존에 `cfg.data.train.pipeline[2]['img_scale'] = (512, 512)`는 `cfg.data.train.pipeline[3]['img_scale'] = (512, 512)` 로 수정해주셔야 합니다.
```python
# 기존 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# 변경된 pipeline
train_pipeline = [
    dict(type='LoadMultiImageFromFile'),
    dict(type='LoadMultiAnnotations', with_bbox=True),
    dict(type='Mosaic'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
```

<br>
<br>

# How to make mix data
- opencv imshow가 필요하기 때문에 local 작업을 추천드립니다.
- 사용법
  - 1 단계 : 해당 폴더로 입력하고 스크립트를 실행합니다.
    ```python
    cd mmdetection_trash
    python class_mix.py
    ``` 
  - 2 단계 : terminal에서 합성할 오브젝트에 대한 클래스 명 혹은 클래스 인덱스를 입력합니다.
  - 3 단계 : opencv window 화면에서 아래 키입력으로 동작을 합니다.
    - `8 4 5 6`: object 상하좌우 이동
    - `7 9` : 상하좌우 이동 스케일 조정
    - `- +` : object scaling
    - `r` : 시계 방향 90도 회전
    - `d f` : background 이미지 변경
    - `c v` : object 변경
    - `s` : 현재 이미지 저장
    - `o` : scale, translation 정보 초기화
    - `a` : 추가 object 선택, 다시 2단계와 같이 terminal에서 추가 할 오브젝트의 클래스 명 혹은 인덱스를 입력합니다.
    - `q` : 합성 종료 및 json 저장 
- args 정보
```python
# data의 root path 입니다.
parser.add_argument('-d','--data_root',              type=str,   help='Root path of json and images', default='input/data') 
# 불러올 json 파일의 이름입니다.
parser.add_argument('-j', '--json_name',              type=str,   help='Name of json file', default='train_all.json') 
# output image의 경로 입니다.
parser.add_argument('-s','--synthesis_folder',              type=str,   help='Folder of result images', default='synthesis_song') 
# output json의 이름입니다.
parser.add_argument('-o','--out_json_name',              type=str,   help='Name of json file', default='train_synthesis_song_all.json')
# background 이미지의 주석 정보를 없애고 싶을 때 flag를 주시면 됩니다. 단일 클래스 예측 모델을 위해 설계된 flag 입니다. 
parser.add_argument('-r','--is_remove',              action='store_true',   help='Remove first background ann') 
# 해당 flag를 주시면 기존 json에 이어쓰지 않고 오직 합성된 정보만 입력된 json이 생성됩니다
parser.add_argument('-n','--is_new',              action='store_true',   help='Remove first background ann') .
```