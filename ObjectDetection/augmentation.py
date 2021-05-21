import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class TrainAugmentation:
    def __init__(self, augp=0.5, resize=(512,512)):
        self.transform = A.Compose([
            A.Resize(resize[0], resize[1]),
            A.Flip(p=augp),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    def __call__(self, image, bboxes, labels):
        return self.transform(image=image, bboxes=bboxes, labels=labels)


class ValidAugmentation:
    def __init__(self, augp=0.5, resize=(512,512)):
        self.transform = A.Compose([
            A.Resize(resize[0], resize[1]),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    def __call__(self, image, bboxes, labels):
        return self.transform(image=image, bboxes=bboxes, labels=labels)

