import torch
import torchvision
import torchvision.transforms as transforms

# the output of dataset in torchvision : [0,1] range , but it seems like the image files are PIL formats
# this is PIL Image which mainly used in Python.
# To open, use Image module. in PIL
# Pillow package support it.


# First, define the transforming function which changes

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Second, call the data

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2)


testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2)
