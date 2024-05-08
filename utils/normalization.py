import numpy as np
import torch

def normalize_embeddings(embeds):
    """
    Normalize the embeddings to the rotation gate format using tensor operations.
    """
    # Convert the labels to a numpy array
    embeds_np = embeds.numpy()

    # Get the unique labels and the inverse indices
    unique_embeds, inverse_indices = np.unique(embeds_np, return_inverse=True)

    # Calculate normalized values and remap according to the original indices
    normalized_embeds = (inverse_indices / len(unique_embeds)) * 2 * np.pi

    return torch.tensor(normalized_embeds, dtype=torch.float32)