import matplotlib.pyplot as plt 
from utils.augmentation import get_train_transforms
import albumentations as A
import torch

def vis_example(image_tensor=None, get_info=False):
    transforms = get_train_transforms()

    for transforms_op in transforms.transforms : 
        if isinstance(transforms_op, A.Normalize):
            mean = torch.tensor(transforms_op.mean)
            std = torch.tensor(transforms_op.std)
            break

    if get_info :
        return mean, std 

    de_nomalized_image = image_tensor * std[:, None, None] + mean[:, None, None]

    de_nomalized_image = de_nomalized_image.permute(1,2,0)

    plt.imshow(de_nomalized_image)
    plt.title("Sample Transformed Image")
    plt.show()

    

    



