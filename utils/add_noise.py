import torch


def add_noise(data, hyperparams):
    """
    Creating T noisy versions of the original data from 0 - T.
    Returns: the T noisy images including the original data.
    """
    # If the data is 1D, add a batch dimension
    if data.dim() == 1: data = data.unsqueeze(0)

    noise = torch.normal(
        mean=hyperparams.mean,
        std=hyperparams.std,
        size=data.shape
    ) 

    # Create a weighting for the noise
    noise_weighting = torch.linspace(0, 1, hyperparams.T + 1)

    # Calculate the weighted components
    data_weighted = data.unsqueeze(0) * (1 - noise_weighting[:, None, None])
    noise_weighted = noise.unsqueeze(0) * noise_weighting[:, None, None]

    # Combine the weighted data and noise
    noisy_data = data_weighted + noise_weighted

    # Rearrange the noisy data dimensions to desired format
    noisy_data = noisy_data.transpose(0, 1)

    return noisy_data
