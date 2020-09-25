import torchvision
# from torch.utils.data import Dataset
import torch


def mnist_dataset(train_transforms, valid_transforms):
    """ Creates MNIST train dataset and a test dataset.
    Args:
    train_transforms: Transforms to be applied to train dataset.
    test_transforms: Transforms to be applied to test dataset.
    """
    # This code can be re-used for other torchvision Image Dataset too.
    train_set = torchvision.datasets.MNIST(
        "./datasets", download=True, train=True, transform=train_transforms
    )

    valid_set = torchvision.datasets.MNIST(
        "./datasets", download=True, train=False, transform=valid_transforms
    )

    return train_set, valid_set


def cifar10_dataset(train_transforms, valid_transforms):
    """ Creates CIFAR10 train dataset and a test dataset.
    Args:
    train_transforms: Transforms to be applied to train dataset.
    test_transforms: Transforms to be applied to test dataset.
    """
    # This code can be re-used for other torchvision Image Dataset too.
    train_set = torchvision.datasets.CIFAR10(
        "./datasets", download=True, train=True, transform=train_transforms
    )

    valid_set = torchvision.datasets.CIFAR10(
        "./datasets", download=True, train=False, transform=valid_transforms
    )

    return train_set, valid_set


def create_loaders(
    train_dataset,
    valid_dataset,
    train_batch_size=32,
    valid_batch_size=32,
    num_workers=1,
    **kwargs
):
    """
    Creates train loader and test loader from train and test datasets
    Args:
    train_dataset: Torchvision train dataset.
    valid_dataset: Torchvision valid dataset.
    train_batch_size (int) : Default 32, Training Batch size
    valid_batch_size (int) : Default 32, Validation Batch size
    num_workers (int) : Defualt 1, Number of workers for training and validation.
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset, train_batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, valid_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader
