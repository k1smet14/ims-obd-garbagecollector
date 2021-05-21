import json
import argparse
from easydict import EasyDict
from importlib import import_module
import numpy as np

from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone#, _validate_trainable_layers
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from scnet import *

class fasterrcnn_resnet50_fpn(torch.nn.Module):
    def __init__(self, num_classes):
        super(fasterrcnn_resnet50_fpn, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, target=None):
        out = self.model(x, target)
        return out

class fasterrcnn_resnet101_fpn(torch.nn.Module):
    def __init__(self, num_classes):
        super(fasterrcnn_resnet101_fpn, self).__init__()
        backbone = resnet_fpn_backbone('resnet101', True)
        self.model = FasterRCNN(backbone, num_classes)
        self.model.roi_heads.score_thresh = args.test_score_threshold

    def forward(self, x, target=None):
        out = self.model(x, target)
        return out

# class fasterrcnn_resnet101_fpn(torch.nn.Module):
#     def __init__(self, num_classes, ):
#         super(fasterrcnn_resnet101_fpn, self).__init__()
#         trainable_backbone_layers = _validate_trainable_layers(
#             pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

#         if pretrained:
#             # no need to download the backbone if pretrained is set
#             pretrained_backbone = False
#         backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
#         model = FasterRCNN(backbone, num_classes, **kwargs)
#         if pretrained:
#             state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
#                                                 progress=progress)
#             model.load_state_dict(state_dict)
#             overwrite_eps(model, 0.0)
#         return model

#     def forward(self, x, target=None):
#         out = self.model(x, target)
#         return out

class fasterrcnn_scnet50_fpn(torch.nn.Module):
    def __init__(self, num_classes, args):
        super(fasterrcnn_scnet50_fpn, self).__init__()
        scnet = scnet50(pretrained=True)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.backbone.body = scnet
        self.model.roi_heads.score_thresh = args.test_score_threshold

        if args.anchor_sizes != None:
            anchor_sizes = tuple((size,) for size in args.anchor_sizes)
            aspect_ratios = (tuple(args.aspect_ratios),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
            self.model.rpn.anchor_generator = rpn_anchor_generator
            # anchor box가 늘어남에 따라, rpn head에서 cls logits와 bbox pred의 output dim을 수정하기 위한 코드
            self.model.rpn.head = RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])


    def forward(self, x, target=None):
        out = self.model(x, target)
        return out

class fasterrcnn_scnet50_fpn_p6p7(torch.nn.Module):
    def __init__(self, num_classes, args):
        super(fasterrcnn_scnet50_fpn_p6p7, self).__init__()
        scnet = scnet50(pretrained=True)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.backbone.body = scnet
        self.model.backbone.fpn.extra_block = LastLevelP6P7(256, 256) 
        self.model.roi_heads.score_thresh = args.test_score_threshold

        if args.anchor_sizes != None:
            anchor_sizes = tuple((size,) for size in args.anchor_sizes)
            aspect_ratios = (tuple(args.aspect_ratios),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
            self.model.rpn.anchor_generator = rpn_anchor_generator
            # anchor box가 늘어남에 따라, rpn head에서 cls logits와 bbox pred의 output dim을 수정하기 위한 코드
            self.model.rpn.head = RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])


    def forward(self, x, target=None):
        out = self.model(x, target)
        return out


class fasterrcnn_scnet101_fpn(torch.nn.Module):
    def __init__(self, num_classes, args):
        super(fasterrcnn_scnet101_fpn, self).__init__()
        scnet = scnet101(pretrained=True)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.backbone.body = scnet
        self.model.roi_heads.score_thresh = args.test_score_threshold

        if args.anchor_sizes != None:
            anchor_sizes = tuple((size,) for size in args.anchor_sizes)
            aspect_ratios = (tuple(args.aspect_ratios),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
            self.model.rpn.anchor_generator = rpn_anchor_generator
            # anchor box가 늘어남에 따라, rpn head에서 cls logits와 bbox pred의 output dim을 수정하기 위한 코드
            self.model.rpn.head = RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])

    def forward(self, x, target=None):
        out = self.model(x, target)
        return out


class fasterrcnn_scnet50_v1d_fpn(torch.nn.Module):
    def __init__(self, num_classes, args):
        super(fasterrcnn_scnet50_v1d_fpn, self).__init__()
        scnet = scnet50_v1d(pretrained=True)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.backbone.body = scnet
        self.model.roi_heads.score_thresh = args.test_score_threshold

        if args.anchor_sizes != None:
            anchor_sizes = tuple((size,) for size in args.anchor_sizes)
            aspect_ratios = (tuple(args.aspect_ratios),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
            self.model.rpn.anchor_generator = rpn_anchor_generator
            # anchor box가 늘어남에 따라, rpn head에서 cls logits와 bbox pred의 output dim을 수정하기 위한 코드
            self.model.rpn.head = RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])

    def forward(self, x, target=None):
        out = self.model(x, target)
        return out



def get_args():
    
    # get config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True) # ex) original_config
    ipts = parser.parse_args()

    # get args in config file
    args = EasyDict()
    with open(f'./config/{ipts.config}.json', 'r') as f:
        args.update(json.load(f))
    
    return args

# checking dataset`s output
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = get_args()

    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes = 11, args=args)
    model.to(device)

    print('* model structure \n', model)
    print('-'*100)

    augmentation_module = getattr(import_module("augmentation"), args.augmentation)
    train_augmentation= augmentation_module(1.0, args.resize)
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(args.annotation, args.data_dir, train_augmentation)
    
    input_data = dataset[0]
    images = [input_data[0].float().to(device)]
    targets = [{k: v.to(device) for k,v in input_data[1].items()}]

    print("* input data \n", images, "\n", targets)
    print('-'*100)

    train_out = model(images, targets)
    model.eval()
    inference_out = model(images)#, targets)

    print('-'*100)
    print("* training output \n", train_out)
    print('-'*100)
    print("* inference output \n", inference_out)

