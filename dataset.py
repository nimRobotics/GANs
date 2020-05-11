import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

class FacesDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.path_files = os.listdir(dataset_dir)

    def __len__(self):
        return len(self.path_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.dataset_dir,self.path_files[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return image
