from configs import Dataset, Hyperparameters, Embeddings
from qdm import QDM
from utils import FID



def parameters():
    dataset = Dataset(
        name="mnist16x16",
        #name="mnist32x32",
        #name="flag_rgb16x16",
        #name="flags_rgb16x16",
        #name="rainbow_rgb16x16",

        labels=[0],
        data_size=1024,
        batch_size=32,
    )

    embeds = Embeddings(
        embeddings=[],
        #embeddings=["timestep"],
        #embeddings=["label"],
        #embeddings=["timestep", "label"],
    )

    hyperparams = Hyperparameters(
        lr=0.001,
        epochs=3,
        qdepth=50,
        T=10,
        mean=0.5,
        std=0.2,
        zeta=0.0075, # suitable for MNIST
        #zeta = 0.02, #rainbow
        #zeta = 0.04 #flag
        sample_zeta_multiplyer=5, # suitable for MNIST
        #sample_zeta_multiplyer=2, #rainbow, flag
        dataset=dataset,
        embeds=embeds
    )

    return dataset, hyperparams, embeds


if __name__ == "__main__":
    # Set the parameters
    dataset, hyperparams, embeds = parameters()

    # Create the model
    qdm = QDM(
        dataset=dataset, 
        hyperparams=hyperparams, 
        embeds=embeds
    )

    # Train the model
    qdm.train_model(
        save_model=True,
    )

    # Test the model
    samples = qdm.sample(
        n=16, 
        save=True,
    )

    if dataset.is_mnist():

        # Calculate the FID score
        fid = FID(
            dataset=dataset, 
            pretrained=True,
        )
        
        score = fid.calculate_fid(samples)
        print(f'FID score: {score}')
