import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class palmPrintDataset(Dataset):
    '''
    示例用法：\\
    dataset = palmPrintDataset(directory='your_dataset_directory') \\
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    '''
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images, self.labels = self.load_data()
        self.classes = np.unique(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return torch.Tensor(image), torch.tensor(label)

    def load_data(self):
        images = []
        labels = []

        # 遍历每个人的文件夹
        for person_folder in os.listdir(self.directory):
            person_path = os.path.join(self.directory, person_folder)
            if os.path.isdir(person_path):
                for file_name in os.listdir(person_path):
                    file_path = os.path.join(person_path, file_name)
                    image = Image.open(file_path)
                    images.append(image)
                    labels.append(int(person_folder)-1)

        return np.array(images), np.array(labels)


