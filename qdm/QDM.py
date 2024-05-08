import torch
from .PQC import PQC
from configs import Dataset, Hyperparameters, Embeddings
from utils import save_samples, add_noise
import numpy as np
from tqdm import tqdm


class QDM(torch.nn.Module):
    """
    Quantum Diffusion Model (QDM) for image denoising.
    """

    def __init__(
        self,
        dataset: Dataset,
        hyperparams: Hyperparameters,
        embeds: Embeddings,
    ) -> None:
        """
        Initializes the QDM class.
        """
        super().__init__()

        self.embeds = embeds
        self.dataset = dataset
        self.hyperparams = hyperparams

        self.net = PQC(
            dataset=dataset,
            hyperparams=hyperparams,
            embeds=embeds,
        )

        self.zetas = []

        if embeds.timestep_embedding():
            self.ts = self.embeds.generate_timesteps(self.hyperparams.T)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """
        x_noisy = add_noise(
            data=x,
            hyperparams=self.hyperparams,
        )

        # All the T noised images
        x_t = x_noisy[:, 1:, :].reshape(
            len(x) * self.hyperparams.T, 
            self.dataset.num_features()
        )

        # The initial image, then the first T-1 noised images
        x_t_minus_1 = x_noisy[:, :-1, :].reshape(
            len(x) * self.hyperparams.T, 
            self.dataset.num_features()
        )

        # Calculate ts_repeated and labels once, outside the conditional logic
        if self.embeds.label_embedding(): labels = np.repeat(y, self.hyperparams.T)
        if self.embeds.timestep_embedding(): ts_repeated = np.repeat(self.ts, len(x))

        # Predict the noise
        predicted_noise = self._predict_noise(
            x=x_t, 
            labels=labels if self.embeds.label_embedding() else None, 
            ts=ts_repeated if self.embeds.timestep_embedding() else None,
        )

        # Calculate the actual noise
        actual_noise = x_t - x_t_minus_1

        return predicted_noise, actual_noise


    def sample(self, n, save=False) -> torch.Tensor:
        """
        Samples n_sample images from the model.

        Args:
            n (int): The number of samples to generate.
            save (bool): Whether to save the samples to /results.
        """
        self.eval()

        # Generate labels
        if self.embeds.label_embedding():
            labels = self.embeds.generate_labels(self.dataset.labels, n)

        # Sample T steps
        with torch.no_grad():

            # Generate random noise
            x = torch.normal(
                mean=self.hyperparams.mean, 
                std=self.hyperparams.std, 
                size=(n, self.dataset.num_features()),
            )

            # Predict the noise
            for i in range(self.hyperparams.T - 1, -1, -1):
                predicted_noise = self._predict_noise(
                    x=x, 
                    labels=labels if self.embeds.label_embedding() else None, 
                    ts=self.ts[i] if self.embeds.timestep_embedding() else None,
                )

                # Remove the noise from the image
                x = x - predicted_noise

        if save:
            save_samples(
                images=x,
                n_samples=n,
                dataset=self.dataset,
                hyperparams=self.hyperparams,
                embeds=self.embeds,
            )

        # Reshape x from (n, num_features) to (n, 1, w, h)
        x = x.view(n, self.dataset.c, self.dataset.h, self.dataset.w)
        
        return x


    def _predict_noise(self, x, labels=None, ts=None):
        """
        Predicts the noise for the given input.

        Args:
            x (torch.Tensor): The input tensor.
            training (bool): Whether the model is training mode.
            labels (torch.Tensor): The labels tensor.
            ts (torch.Tensor): The timesteps tensor.
        """
        predicted_noise = self.net(
            inp=x,
            labels=labels,
            timesteps=ts,
        )

        # Sum the noise
        sums = predicted_noise.sum(dim=1)

        # Subtract the mean from the noise
        predicted_noise -= (sums / self.dataset.num_features()).view(-1, 1)

        # Multiply the noise by zeta
        if self.training: predicted_noise *= self.hyperparams.zeta
        else: predicted_noise *= self.hyperparams.sample_zeta

        return predicted_noise
    

    def train_model(self, save_model=False):
        """
        Trains the model.

        Args:
            save_model (bool): Whether to save the model to "model.pth".
        """
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hyperparams.lr)

        self.train()

        for _ in tqdm(range(self.hyperparams.epochs), desc="Epoch"):
            for x, y in tqdm(self.dataset.dataloader, desc="Data", leave=False):
                optimizer.zero_grad()

                predicted_noise, actual_noise = self(x, y)

                loss = loss_func(predicted_noise, actual_noise)
                loss.backward()

                optimizer.step()

        if save_model:
            torch.save(self.state_dict(), "model.pth")