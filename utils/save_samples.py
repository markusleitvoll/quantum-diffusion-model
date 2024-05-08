import torch
import math
from torchvision.utils import make_grid, save_image
import os

def rgb_reshape_for_plotting(image, dataset):
    # Assuming 'data' is a numpy array or a similar iterable with 192 elements (64 R, 64 G, 64 B)
    # Split the data into R, G, and B components

    R = image[:dataset.num_pixels()]
    G = image[dataset.num_pixels():dataset.num_pixels()*2]
    B = image[dataset.num_pixels()*2:]

    shape = (dataset.w, dataset.h, dataset.c)

    # Initialize an empty array for the reshaped data
    reshaped_data = torch.zeros(shape[0], shape[1], 3)

    # Fill in the reshaped data with R, G, and B values
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = i * shape[0] + j
            reshaped_data[i, j, 0] = R[index]  # R value
            reshaped_data[i, j, 1] = G[index]  # G value
            reshaped_data[i, j, 2] = B[index]  # B value

    return reshaped_data


def save_samples(images, n_samples, dataset, hyperparams, embeds):

    if dataset.is_rgb():
        results = []

        for i in range(images.size(0)):
            result = rgb_reshape_for_plotting(images[i], dataset)
            results.append(result)

        # Stack the resulting tensors to get a final tensor of shape
        result_tensor = torch.stack(results)

        # Transpose the array to have channels as the second dimension
        result_tensor = result_tensor.permute(0, 3, 1, 2)

    else:
        min_values = torch.min(images, dim=1, keepdim=True).values
        max_values = torch.max(images, dim=1, keepdim=True).values
        result_tensor = (images - min_values) / (max_values - min_values)

        result_tensor = result_tensor.reshape(n_samples, 1, int(math.sqrt(dataset.num_pixels())), int(math.sqrt(dataset.num_pixels())))
        
        result_tensor = 1 - result_tensor

    os.makedirs(os.path.dirname(f'./results/{dataset.name}/{embeds.get_folder_name()}/{hyperparams.get_filename()}'), exist_ok=True)
    grid = make_grid(result_tensor[:16], nrow=int(math.sqrt(16)))
    save_image(grid, f'./results/{dataset.name}/{embeds.get_folder_name()}/{hyperparams.get_filename()}.png')