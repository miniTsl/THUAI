import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from models.vit import ViT
from models.Resnet34 import ResNet, BasicBlock
from models.basic_CNN import basic_CNN
from models.SVM import SVM
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ViT", type=str, help="[ViT, basic_cnn, resnet, SVM]")
parser.add_argument("--svm-kernel", default="linear", type=str, help="[linear, rbf, poly]")
parser.add_argument("--weights", default="vit_c10_aa_ls", type=str)
args = parser.parse_args()

# randomly choose one picture from CIFAR-1O test dataset
def choose_picture():
    # download=True，back-up，then download=False
    # 跑demo的机器上最好先下一份数据，这样直接就能run了
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    # randomly choose one picture
    random_index = torch.randint(0, len(test_dataset), (1,))
    picture, label = test_dataset[random_index]
    return picture, label

def plot(model_name, model, test_transform):
    label_text = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(15, 5))
    plt.suptitle(model_name + " Prediction Result" + " (true-prediction)", fontsize=15)
    for i in range(3):
        picture, label = choose_picture()
        img_plt = np.array(picture)

        if model_name == "ViT": # vit needs batch dimension
            picture = test_transform(picture)
            picture = picture.unsqueeze(0)
            prediction = model(picture).argmax(-1)
        elif model_name == "resnet":
            picture = test_transform(picture)
            prediction = model(picture.unsqueeze(0)).argmax(-1)
        elif model_name == 'basic_cnn':
            picture = test_transform(picture)
            prediction = model(picture.unsqueeze(0)).argmax(-1)
        elif model_name == 'SVM':
            picture =  picture.numpy().reshape(1, -1)
            prediction = model.predict(picture)
        else:
            prediction = None

        true_label = label_text[label]
        prediction_label = label_text[prediction]
        plt.subplot(1, 3, i+1)
        plt.imshow(img_plt)
        plt.title(true_label + "-" + prediction_label, fontsize=15)
        plt.axis('off')
    plt.savefig("./demo/demo1.png")


if __name__ == '__main__':
    model_name = args.model #"ViT" or "SVM", "resnet", "basic cnn"
    weight_path = "weights/"+args.weights+'.pth'

    if model_name == "ViT":
        model = ViT(head=12)
        state_dict = torch.load(weight_path)
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}    # 保存权重时多了个model的前缀
        model.load_state_dict(new_state_dict)
        
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]  # CIFAR-10
        test_transform = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)] 
        test_transform = transforms.Compose(test_transform)
        
    elif model_name == "resnet":
        model = ResNet(BasicBlock, 3, [3, 4, 6, 3])
        state_dict = torch.load(weight_path)
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        test_transform = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        test_transform = transforms.Compose(test_transform)

    elif model_name == "basic_cnn":
        model = basic_CNN(3, 10)
        state_dict = torch.load(weight_path)
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        test_transform = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        test_transform = transforms.Compose(test_transform)

    elif model_name == 'SVM':
        model = SVM(kernel=args.svm_kernel)
        with open(weight_path, 'rb') as f:
            model = pickle.load(f)
        test_transform = None

    else:
        model = None
        test_transform = None
        
    plot(model_name, model, test_transform)