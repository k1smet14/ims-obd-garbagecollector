import albumentations as A 
from albumentations.pytorch import ToTensorV2


class CustomAugmentation:
    def __init__(self, augp=0.5, resize=(512,512)):
        self.transform = A.Compose([
            A.Resize(resize[0], resize[1], p=1.0),
            A.HorizontalFlip(p=augp),
            A.ShiftScaleRotate(p=augp),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=(-0.1, 0.3), p=augp),
            A.GaussNoise(p=augp),
            ToTensorV2(p=1.0)
        ], p=1.0)
    
    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)


class ValAugmentation:
    def __init__(self, resize=(512,512)):
        self.transform = A.Compose([
            A.Resize(resize[0], resize[1], p=1.0),
            ToTensorV2(p=1.0)
        ], p=1.0)
    
    def __call__(self, image, mask=False):
        return self.transform(image=image, mask=mask)


class TestAugmentation:
    def __init__(self, resize=(512,512)):
        self.transform = A.Compose([
            A.Resize(resize[0], resize[1], p=1.0),
            ToTensorV2(p=1.0)
        ], p=1.0)
    
    def __call__(self, image):
        return self.transform(image=image)
