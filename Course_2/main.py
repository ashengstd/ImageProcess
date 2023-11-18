import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models.resnet import ResNetClassifier
from models.vgg import VGGClassifier
from torch.utils.data import DataLoader,random_split
from utils.dataset import palmPrintDataset
import argparse

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义命令行参数
parser = argparse.ArgumentParser(description='CNN Model Selection')
parser.add_argument('--model', type=str, default='vgg', choices=['vgg', 'resnet'],
                    help='Choose the CNN model (vgg or resnet)')
args = parser.parse_args()

# 定义数据预处理和加载
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

# 替换为你的数据集路径和类别数
dataset = palmPrintDataset(directory='output', transform=transform)
total_size = len(dataset)
train_size = int(0.5 * total_size)
train_dataset, test_dataset = random_split(dataset, [train_size, total_size - train_size])

# 创建用于训练和测试的数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 选择模型
if args.model == 'vgg':
    model = VGGClassifier(num_classes=len(dataset.classes), weights='DEFAULT').to(device)
    num_epochs = 20
elif args.model == 'resnet':
    model = ResNetClassifier(num_classes=len(dataset.classes), weights='DEFAULT').to(device)
    num_epochs = 10
else:
    raise ValueError('Invalid model choice. Please choose between "vgg" and "resnet".')

# 初始化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将数据移动到GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}')

# 测试模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将数据移动到GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on the test set: {accuracy:.2f}')
