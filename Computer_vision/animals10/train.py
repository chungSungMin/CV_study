import os 
import torch 
import torch.nn as nn
from utils.dataset import get_dataset, get_dataloader
from utils.visualize import vis_example
from utils.get_device import set_device
from utils.set_seed import set_seed
from models.resnet34 import get_model
from tqdm import tqdm


def start_train(train_dataloader, val_dataloader, period, epochs, num_class, seed):
    set_seed(seed)
    device = set_device()
    model = get_model(num_class).to(device)
    
    min_val_loss = float("inf")

    # ckpt_path = os.path.join(os.getcwd(),"models/ckpt")
    # if os.listdir(ckpt_path):
    #     print(f"사전 학습 가중치가 존재하여 모델의 가중치를 저장합니다.")
    #     ckpt = "/Users/jeongseungmin/Desktop/Study/CV_parctice/animals/models/ckpt/animals_finetuned.pth"

    #     state_dict = torch.load(ckpt, map_location=device) # ckpt에 저장된 값들을 딕셔너리 형태로 가져오고
    #     model.load_state_dict(state_dict) # 가져온 딕셔너리를 모델에 삽입한다

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs) :
        print(f"[{epoch+1} | {epochs}] Training...🐎🐴🎠🐎🐴🎠🐎🐴🎠")
        model.train()
        running_loss = 0.0

        for img, label in tqdm(train_dataloader) :
            img, labels = img.to(device), label.to(device)

            optimizer.zero_grad()

            outputs = model(img) # [32, 10] 크기를 갖는 Tensor() 이다. 

            loss = criterion(outputs, labels) # 손실의 평균을 반환한다.

            """loss.backward()
            optimzer.zero_gard()시 기울기 값들이 None으로 초기화가 되고, 이후 loss.backward()를 진행하게 되면 
            파라미터와 동일한 크기의 Tensor에 누적된 기울기 값들이 쌓이게 됩니다.
            배치사이즈의 크기만큼의 데이터에서 나온 기울기를 누적합니다. 즉, 배치사이즈 크기만큼의 기울기 평균값을 얻게 됩니다.
            """
            # print(f"Before loss.backward() : {model.fc.weight.grad}")
            loss.backward()
            # print(f"After loss.backward() : {model.fc.weight.grad}")
            # print(f"Shape of Gradient : {model.fc.weight.grad.shape}")

            optimizer.step() # 위에서 구한 기울기를 기반으로 가중치를 업데이트 한다.

            running_loss += loss.item() * img.size(0) # loss의 경우 0차원 텐서이기에 이를 Float으로 사용하기 위해서 item()을 사용, img.size(0) = 배치사이즈를 곱해서 평균을 통해 전체 손실을 복원

        
        epoch_loss = running_loss / len(train_dataloader.dataset) # running_loss에 배치별 손실이 싸이고, 전체 데이터를 다 본 후에는 전체 데이터로 모든 배치의 손실을 평균해줘서. 결국 1에폭의 평균을 계산해준다.
        print(f"[{epoch+1}/{epochs}] Epoch Loss : {epoch_loss: .5f}")

        if (epoch + 1) % val_period == 0 :
            model.eval()
            val_loss = 0.0
            total = 0
            correct = 0
            
            with torch.no_grad():
                for img, label in val_dataloader:
                    img, label = img.to(device), label.to(device)

                    outputs = model(img)
                    loss = criterion(outputs, label)

                    val_loss += loss.item() * img.size(0)
                    # valeus , predicted = torch.max(otuputs, 1) # 여기서 그래서 보통 values는 _로 처리한다.
                    predicted = outputs.argmax(dim=1)

                    total += label.size(0)

                    correct += (predicted==label).sum().item()
                print(f"Validation Accuracy : {100 * correct / total: .5f}")       

                if val_loss / total <= min_val_loss : 
                    min_val_loss = val_loss / total
                    print("새로운 모델 가중치를 저장합니다.")
                    save_path = os.path.join(os.getcwd(),"fintuned_resnet34.pth")
                    torch.save(model.state_dict(), save_path)  

    return model


if __name__ == "__main__" : 
    # pwd = os.getcwd()
    # data_path = os.path.join(pwd, "data")

    data_path = "/root/.cache/kagglehub/datasets/alessiocorrado99/animals10/versions/2/raw-img"

    train_dataset, val_dataset, test_dataset = get_dataset(data_path)

    train_dataloader, val_dataloader , test_dataloader = get_dataloader(
        train_dataset, val_dataset, test_dataset
    )

    epoch = 3
    val_period = 10
    trained_model = start_train(train_dataloader,val_dataloader, val_period, epoch, num_class=10, seed=42)

    # example_image = train_data[0][0]
    # vis_example(example_image)










