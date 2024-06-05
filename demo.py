import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

# randomly choose one picture from CIFAR-1O test dataset
def choose_picture(transform=None):
    # 加载数据集，这个我们本地下一份就好了，可以先download=True，然后再download=False
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    # randomly choose one picture
    random_index = torch.randint(0, len(test_dataset), (1,))
    picture, label = test_dataset[random_index]
    return picture, label

model = None


def plot(model, transform):
    plt.figure(figsize=(15, 5))
    for i in range(3):
        picture, label = choose_picture()
        # 定义转换操作，需要将PIL Image转换为PyTorch张量
        transform = transforms.ToTensor()
        img_tensor = transform(picture)
        # 将张量转换为NumPy数组，并改变维度顺序以适应Matplotlib
        img_np = img_tensor.numpy().transpose((1, 2, 0))
        # 将label转换为文本
        label_text = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        label = label_text[label]
        
        prediction = model(img_tensor.unsqueeze(0))
        
        plt.subplot(1, 3, i+1)
        plt.imshow(img_np)
        plt.title(label)
        plt.axis('off')
    
model_text = "ViT" # or "SVM", "ResNet"
plt.title(model_text + " Prediction Result", fontsize=15)
plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.title(label, fontsize=15)
plt.axis('off')
plt.subplot(1, 2, 2)



text = "The model is " + model_text + "\n" + ""
plt.text(0.5, 0.5, 'This is xxx', fontsize=15, ha='center')
plt.axis('off')
plt.title('Prediction Result', fontsize=15)
plt.savefig('result.png')



if __name__ == '__main__':
    transform = transforms.ToTensor()
    plot(model, transform)