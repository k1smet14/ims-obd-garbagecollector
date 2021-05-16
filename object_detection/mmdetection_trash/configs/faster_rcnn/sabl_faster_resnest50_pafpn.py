_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../trash/dataset.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnest50',
    backbone=dict(
        type='ResNeSt',
        stem_channels=64,
        depth=50,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(
            _delete_=True,
            type='SABLHead',
            num_classes=11,
            cls_in_channels=256,
            reg_in_channels=256,
            roi_feat_size=7,
            reg_feat_up_ratio=2,
            reg_pre_kernel=3,
            reg_post_kernel=3,
            reg_pre_num=2,
            reg_post_num=1,
            cls_out_channels=1024,
            reg_offset_out_channels=256,
            reg_cls_out_channels=256,
            num_cls_fcs=1,
            num_reg_fcs=0,
            reg_class_agnostic=True,
            norm_cfg=None,
            bbox_coder=dict(
                type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.7),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1,
                               loss_weight=1.0)))
)

# # use ResNeSt img_norm
img_norm_cfg = dict(mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=False,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))