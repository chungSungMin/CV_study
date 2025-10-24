
from utils.dataset import get_dataset, get_dataloader
from utils.get_device import set_device
from models.resnet18 import get_model
from utils.set_seed import set_seed
from utils.visualize import vis_example
import os 
import torch 
import torch.nn as nn 
import numpy as np 

import matplotlib.pyplot as plt 
from pytorch_grad_cam import GradCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def run_gradcam_visualization(test_dataloader, num_class, seed, path, num_images=5):
    device = set_device()
    set_seed(seed)
    model = get_model(num_class)

    stated_dict = torch.load("/workspace/CV_study/Computer_vision/animals10/fintuned_resnet34.pth", map_location=device)
    model.load_state_dict(stated_dict)
    model.eval()
    model.to(device)
    
    target_layer = [model.layer4[-1]]

    # cam = GradCAM(model=model, target_layers=target_layer)
    cam = AblationCAM(model=model, target_layers=target_layer)

    images_shown = 0
    mean, std = vis_example(get_info=True)
    mean, std = mean.cpu().numpy(), std.cpu().numpy()

    for img_batch, label_batch in test_dataloader : 
        if images_shown >= num_images : 
            print("시각화 완료 ")
            break

        img_np = img_batch[0].permute(1,2,0).cpu().numpy()

        img_np = std * img_np + mean   
        img_np = np.clip(img_np, 0, 1) 

        input_tensor = img_batch[0].unsqueeze(0).to(device)
        true_label = label_batch[0].to(device)

        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
        predicted_class = predicted_class.item()

        targets = [ClassifierOutputTarget(predicted_class)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title(f"Original Image\nTrue: {true_label}, Pred: {predicted_class}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.title("Grad-CAM")
        plt.axis('off')
        
        save_path = f"visualization_{images_shown}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved")

        images_shown += 1





def start_predict(test_dataloader, num_class, seed, path):
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
    data_path = "/root/.cache/kagglehub/datasets/alessiocorrado99/animals10/versions/2/raw-img"


    train_dataset, val_dataset, test_dataset = get_dataset(data_path)

    _, _ , test_dataloader = get_dataloader(
        train_dataset, val_dataset, test_dataset
    )

    # start_predict(
    #     test_dataloader=test_dataloader, 
    #     num_class=10, 
    #     seed=42, 
    #     path="/workspace/fintuned_resnet34.pth"
    #     )

    run_gradcam_visualization(
        test_dataloader=test_dataloader, 
        num_class=10, 
        seed=42, 
        path="/workspace/fintuned_resnet34.pth",
        num_images=5
    )