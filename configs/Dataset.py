from dataclasses import dataclass, field
from typing import Any
import math
import torch
from torchvision import datasets as torchvision_datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from utils.normalization import normalize_embeddings


@dataclass
class Dataset:
    """
    A class to represent a dataset.

    Args:
        name (str): The name of the dataset.
        labels (list): The labels to use.
        data_size (int): The size of the dataset.
        batch_size (int): The size of the batches.
        dataloader (Any): The DataLoader object.
        w (int): The width of the images.
        h (int): The height of the images.
        c (int): The number of channels of the images.
    """

    name: str
    labels: list
    data_size: int
    batch_size: int
    dataloader: DataLoader = field(init=False)
    w: int = field(init=False)
    h: int = field(init=False)
    c: int = field(init=False)

    def __post_init__(self):
        dataset_funcs = {
            "mnist16x16": lambda: self._load_mnist((16, 16, 1)),
            "mnist32x32": lambda: self._load_mnist((32, 32, 1)),
            "flag_rgb16x16": lambda: self._load_image_folder("data/flag", (16, 16, 3)),
            "flags_rgb16x16": lambda: self._load_image_folder("data/flags", (16, 16, 3)),
            "rainbow_rgb16x16": lambda: self._load_image_folder("data/rainbow", (16, 16, 3)),
        }

        if self.name not in dataset_funcs:
            raise ValueError(f"Dataset {self.name} not supported")

        # Load the dataset
        self.dataloader, (self.w, self.h, self.c) = dataset_funcs[self.name]()

    def _load_image_folder(self, root, dimensions):
        """
        Load an ImageFolder dataset and return a DataLoader with the images and labels.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(dimensions[:2], antialias=True),
            Dataset._flatten_img,
        ])

        # Load the ImageFolder dataset
        dataset = torchvision_datasets.ImageFolder(root=root, transform=transform)

        # Extract images and labels from the dataset
        images, labels = zip(*[(img, label) for img, label in dataset])

        # Convert lists to PyTorch tensors
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        # Normalize labels
        labels_tensor = normalize_embeddings(labels_tensor)

        # Create a TensorDataset with your images and processed labels
        dataset = TensorDataset(images_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader, dimensions

    def _load_mnist(self, dimensions):
        """
        Load the MNIST dataset and return a DataLoader with the images and labels.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(dimensions[:2], antialias=True),
        ])

        # Load the MNIST dataset
        ds = torchvision_datasets.MNIST(
            root="~/mnist", download=True, transform=transform
        )

        # Filter the dataset by the labels and apply the mask
        mask = (ds.targets.unsqueeze(1) == torch.tensor(self.labels)).any(1)
        ds.data = ds.data[mask]
        ds.targets = ds.targets[mask]

        # Extract images and labels from the dataset
        x_train, y_train = (
            DataLoader(ds, batch_size=self.data_size, shuffle=False)
            .__iter__()
            .__next__()
        )

        # Flatten the images
        x_train = x_train.flatten(start_dim=1)

        # Invert the images
        x_train = 1 - x_train

        # Convert the images and labels to double
        x_train = x_train.to(torch.double)
        y_train = y_train.to(torch.double)

        # Normalize the labels
        y_train = normalize_embeddings(y_train)

        # Create a TensorDataset with the images and processed labels
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return dataloader, dimensions

    @staticmethod
    def _flatten_img(tensor):
        return torch.flatten(tensor).double()

    def num_pixels(self):
        return self.w * self.h

    def num_features(self):
        return self.w * self.h * self.c

    def num_wires(self):
        return math.ceil(math.log2(self.w * self.h * self.c))

    def is_rgb(self):
        return self.c == 3
    
    def is_mnist(self):
        return "mnist" in self.name
    
    def fid_preprocessing(self):
        """
        Preprocess the dataset for FID calculation.
        """
        imgs = [b[0].view(-1, self.c, self.w, self.h) for b in self.dataloader]
        return torch.cat(imgs, dim=0)
