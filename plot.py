import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 读取上传的CSV文件
file_path = 'THUAI/logs/metrics.csv'
data = pd.read_csv(file_path)

# 提取数据：根据列名中的非NaN值来提取相应的行数据
train_loss_data = data.dropna(subset=['loss'])
val_loss_data = data.dropna(subset=['val_loss'])
train_acc_data = data.dropna(subset=['acc'])
val_acc_data = data.dropna(subset=['val_acc'])
lr_data = data.dropna(subset=['lr'])

# 绘制图1和图2：训练损失和验证损失，训练准确率和验证准确率
plt.figure(figsize=(12, 6))

# 子图1：训练损失和验证损失
plt.subplot(2, 1, 1)
plt.plot(train_loss_data['step'], train_loss_data['loss'], label='Train Loss', alpha=0.7)
plt.plot(val_loss_data['step'], val_loss_data['val_loss'], label='Validation Loss', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Train Loss and Validation Loss')
plt.legend()
plt.grid(True)


# 子图2：训练准确率和验证准确率
plt.subplot(2, 1, 2)
plt.plot(train_acc_data['step'], train_acc_data['acc'], label='Train Accuracy', alpha=0.7)
plt.plot(val_acc_data['step'], val_acc_data['val_acc'], label='Validation Accuracy', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Train Accuracy and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))

plt.tight_layout()
plt.show()

# 绘制图3：学习率的变化
plt.figure(figsize=(12, 6))
plt.plot(lr_data['step'], lr_data['lr'], label='Learning Rate', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
plt.grid(True)
plt.show()
