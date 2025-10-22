import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

def get_train_transforms():
    alb_transforms = A.Compose([
        A.Resize(224,224),
        A.HorizontalFlip(),
        A.Rotate(limit=90, p=1, border_mode=cv2.BORDER_REPLICATE), #  border_mode의 경우 회전되어 생긴 빈 공간을 어떻게 채울지에 대한 모드이다.
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return alb_transforms


def get_val_transforms():
    alb_transforms = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return alb_transforms


def apply_alb_transforms(pil_image, transforms):
    numpy_iamge = np.array(pil_image)
    transformed = transforms(image=numpy_iamge)

    return transformed["image"]





