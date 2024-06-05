import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import transforms


# randomly choose one picture from CIFAR-1O test dataset
def choose_picture(transform=None):
    # load CIFAR-10 test dataset
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    # randomly choose one picture
    random_index = torch.randint(0, len(test_dataset), (1,))
    picture, label = test_dataset[random_index]
    return picture, label

picture, label = choose_picture()
# 定义转换操作，将PIL Image转换为PyTorch张量
transform = transforms.ToTensor()

# 将PIL Image转换为PyTorch张量
img_tensor = transform(picture)

# 将张量转换为NumPy数组，并改变维度顺序以适应Matplotlib
img_np = img_tensor.numpy().transpose((1, 2, 0))

# 显示图像
plt.imshow(img_np)
plt.axis('off')  # 关闭坐标轴
plt.savefig('test.png')

