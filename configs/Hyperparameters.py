from dataclasses import InitVar, dataclass, field
from configs import Dataset, Embeddings

@dataclass
class Hyperparameters:
    """
    A class to represent the hyperparameters of the model.
    
    Args:
        lr (float): The learning rate.
        epochs (int): The number of epochs.
        qdepth (int): The depth of the quantum circuit.
        T (int): The number of noise steps.
        zeta (float): The noise amplification factor.
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.
    """
    lr: float
    epochs: int
    qdepth: int
    T: int
    mean: float
    std: float
    zeta: float
    sample_zeta_multiplyer: float
    sample_zeta: float = field(init=False)
    dataset: InitVar[Dataset]
    embeds: InitVar[Embeddings]

    def __post_init__(self, dataset: Dataset, embeds: Embeddings):
        
        # Increase zeta if the dataset is RGB
        #if dataset.c == 3: self.zeta *= 1.33

        # Increase zeta based on the number of features
        self.zeta *= dataset.num_features()

        # Increase zeta based on the number of ancilla qubits
        if embeds.num_ancilla() > 0: self.zeta *= 2 * embeds.num_ancilla()

        # Set the sample zeta
        self.sample_zeta = self.zeta * self.sample_zeta_multiplyer
        
        # Set labels for filename
        self.labels = dataset.labels

    def get_filename(self):
        return f'lr{self.lr}_z{self.zeta}_sz{self.sample_zeta}_t{self.T}_e{self.epochs}_q{self.qdepth}_m{self.mean}_s{self.std}_{self.labels}'