import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models.resnet import ResNetClassifier
from models.vgg import VGGClassifier
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import palmPrintDataset
import argparse
from tqdm import tqdm

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义命令行参数
parser = argparse.ArgumentParser(description='CNN Model Selection')
parser.add_argument('--model', '-m', type=str, default='vgg', choices=['vgg', 'resnet'],
                    help='Choose the CNN model (vgg or resnet)')
parser.add_argument('--directory', '-d', type=str, default='output',
                    help='Where the dataset is saved')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of epochs to train the model')
parser.add_argument('-t', '--tensorboard', action='store_true', 
                    help='Enable TensorBoard logging')
args = parser.parse_args()

print('Running model selection with the following parameters:')
print(f'Model: {args.model}')
print(f'Dataset directory: {args.directory}')

# 定义数据预处理和加载
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

# 替换为你的数据集路径和类别数
dataset = palmPrintDataset(directory=args.directory, transform=transform)
total_size = len(dataset)
train_size = int(0.5 * total_size)
train_dataset, test_dataset = random_split(dataset, [train_size, total_size - train_size])

# 创建用于训练和测试的数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 选择模型
if args.model == 'vgg':
    model = VGGClassifier(num_classes=len(dataset.classes), weights='DEFAULT').to(device)
elif args.model == 'resnet':
    model = ResNetClassifier(num_classes=len(dataset.classes), weights='DEFAULT').to(device)
else:
    raise ValueError('Invalid model choice. Please choose between "vgg" and "resnet".')

if args.tensorboard:
    writer = SummaryWriter(log_dir='logs')
else:
    writer = None

# 初始化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(args.epoch):
    model.train()
    total_loss = 0.0

    # 使用tqdm显示训练进度
    for inputs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.epoch}',leave=False):
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将数据移动到GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{args.epoch}, Loss: {total_loss / len(train_dataloader):.6f}')

    # 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc='Testing',leave=False):
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将数据移动到GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_test_loss = loss / len(test_dataloader)
    print(f'Test, Loss: {total_loss / len(train_dataloader):.6f}, Test Acc: {accuracy:.4f}')

    # 写入Tensorboard
    if writer is not None:
        writer.add_scalar('Training/Loss', avg_train_loss, epoch + 1)
        writer.add_scalar('Testing/Loss', avg_test_loss, epoch + 1)
        writer.add_scalar('Testing/Accuracy', accuracy, epoch + 1)

# 关闭Tensorboard
if writer is not None:
    writer.close()

# 测试模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_dataloader, desc='Testing',leave=False):
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将数据移动到GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Final Accuracy on the test set: {accuracy:.4f}')
