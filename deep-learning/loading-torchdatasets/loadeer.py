import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transformations (convert to tensor + normalize)
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))   # normalize to [-1,1]
])

# Download and load training data
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)

# Download and load test data
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    transform=transform,
    download=True
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check sample batch
images, labels = next(iter(train_loader))
print("Batch of images shape:", images.shape)
print("Batch of labels shape:", labels.shape)
