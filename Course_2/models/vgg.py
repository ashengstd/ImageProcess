import torch.nn as nn
import torchvision.models as models

class VGGClassifier(nn.Module):
    def __init__(self, num_classes,weights='DEFAULT',channels=1):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16_bn(weights=weights)  # 使用VGG16作为基础模型
        self.vgg.features[0] = nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)