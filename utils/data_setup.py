from utils.data_loader import CustomDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F


## Transforms
train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # we can add normalization here
])
test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

## datasets
train_folder = "./merged/train"
valid_folder = "./merged/valid"
test_folder = "./merged/test"

train_dataset = CustomDataset(train_folder,transform=train_transform)
valid_dataset = CustomDataset(train_folder,transform=test_transform)
test_dataset = CustomDataset(train_folder,transform=test_transform)

## Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



