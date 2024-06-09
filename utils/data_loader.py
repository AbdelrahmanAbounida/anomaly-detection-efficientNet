from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self,data_dir:str,transform=None):
        self.data = ImageFolder(data_dir,transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index:int):
        if index < 0 or index > len(self.data):
            return
        return self.data[index]

    @property
    def classes(self):
        return self.data.classes
    