
from utils.dataset import get_dataset, get_dataloader
from utils.get_device import set_device
from models.resnet34 import get_model
from utils.set_seed import set_seed
import os 
import torch 
import torch.nn as nn 


def test(test_dataloader, num_class, seed, path):
    device = set_device()
    set_seed(seed)
    model = get_model(num_class)
    criterion = nn.CrossEntropyLoss()

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    with torch.no_grad():
        total = 0
        correct = 0
        test_loss = 0.0
        for img, label in test_dataloader:
            img, label = img.to(device), label.to(device)

            outputs = model(img)
            loss = criterion(outputs, label)

            test_loss += loss.item() * img.size(0)
            _, predicted = torch.max(outputs, 1)

            total += img.size(0)
            correct += (predicted==label).sum().item()
        print(f"Test Accuracy : {100 * correct / total :.4f}")
        print(f"Average Loss : {test_loss / total :.4f}")




if __name__ == "__main__" : 

    # pwd = os.getcwd()
    # data_path = os.path.join(pwd, "data")
    data_path = "/root/.cache/kagglehub/datasets/alessiocorrado99/animals10/versions/2"


    train_dataset, val_dataset, test_dataset = get_dataset(data_path)

    _, _ , test_dataloader = get_dataloader(
        train_dataset, val_dataset, test_dataset
    )

    test(
        test_dataloader=test_dataloader, 
        num_class=10, 
        seed=42, 
        path="/workspace/fintuned_resnet34.pth"
        )