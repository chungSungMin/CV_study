import torchvision.models as models
import torch.nn as nn

def get_model(num_class):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, num_class)

    return model

if __name__ == "__main__":
    get_model(10)
