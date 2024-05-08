import torch
from torch import nn
from configs import Dataset
from scipy.linalg import sqrtm
import numpy as np
import torchvision.transforms.functional as TF


class FID(nn.Module):
    def __init__(self, dataset: Dataset, pretrained=True):
        super(FID, self).__init__()

        self.dataset = dataset

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        if pretrained:
            self.load_state_dict(torch.load("./models/fe_model.pth"))
        else:
            self.train_model()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 32 * 4 * 4)
        features = self.fc_layers(x)
        return features

    def train_model(self, dataloader, epochs=100, save_model=False):
        """
        Trains the model.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use for training.
            epochs (int): The number of epochs to train for.
            save_model (bool): Whether to save the model to "fe_model.pth".
        """
        self.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())

        for _ in range(epochs):
            for images, labels in dataloader:
                images = images.view(100, 16, 16)
                images = images.unsqueeze(1)
                images = images.float()

                optimizer.zero_grad()
                output = self(images)
                loss = criterion(output.squeeze(), labels)
                loss.backward()
                optimizer.step()

        if save_model:
            torch.save(self.state_dict(), "./models/fe_model.pth")

    def calculate_fid(self, samples):
        self.eval()

        # Preprocess the real images
        real = self.dataset.fid_preprocessing().float()
        real = torch.stack([TF.resize(img, [112, 112], antialias=True) for img in real])

        # Ensure the samples are also preprocessed accordingly
        samples = samples.float()
        samples = torch.stack([TF.resize(img, [112, 112], antialias=True) for img in samples])

        # Calculate activations without tracking gradients
        with torch.no_grad():
            act1 = self(real).detach().cpu().numpy()
            act2 = self(samples).detach().cpu().numpy()

        # Calculate the mean and covariance for the activations
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        # Compute the squared difference of the means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)

        # Compute the product of covariances and its square root
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):  # Correcting possible imaginary numbers
            covmean = covmean.real

        # Calculate the FID score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid