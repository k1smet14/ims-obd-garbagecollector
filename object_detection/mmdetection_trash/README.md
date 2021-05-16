## How to use Mosaic 
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