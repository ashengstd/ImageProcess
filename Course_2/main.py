import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models.resnet import ResNetClassifier
from torch.utils.data import DataLoader
from utils.dataset import palmPrintDataset  # 替换为你的数据集类

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理和加载
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])


# 替换为你的数据集路径和类别数
dataset = palmPrintDataset(directory='output', transform=transform)
resnet_model = ResNetClassifier(num_classes=len(dataset.classes), weights='DEFAULT').to(device)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
# 初始化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5

for epoch in range(num_epochs):
    resnet_model.train()
    total_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将数据移动到GPU
        optimizer.zero_grad()
        outputs = resnet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')

# 测试模型
resnet_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将数据移动到GPU
        outputs = resnet_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on the test set: {accuracy:.2f}')
