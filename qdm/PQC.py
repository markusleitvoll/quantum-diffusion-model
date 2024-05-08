import torch
import pennylane as qml
import qw_map
from configs import Dataset, Hyperparameters, Embeddings


class PQC(torch.nn.Module):
    """
    Parametrized Quantum Circuit (PQC) model for image classification.
    """

    def __init__(
        self,
        dataset: Dataset,
        hyperparams: Hyperparameters,
        embeds: Embeddings,
    ) -> None:
        """
        Initializes the PQC class.

        Args:
            embeds (ModelEmbeddings): The model embeddings (label and timestep)
            dataset (Dataset): The dataset
            hyperparams (Hyperparameters): The hyperparameters

        Returns:
            None
        """
        super().__init__()

        self.embeds = embeds
        self.dataset = dataset
        self.hyperparams = hyperparams
        self.wires = dataset.num_wires() + embeds.num_ancilla()
        self.device = qml.device("default.qubit.torch", wires=self.wires)
        
        # Initialize the weights of the quantum circuit
        weights_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.hyperparams.qdepth,
            n_wires=self.wires,
        )
        self.weights = torch.nn.Parameter(
            torch.randn(weights_shape, requires_grad=True)
        )

        # Define the quantum node
        self.qnode = qml.QNode(
            func=self.circuit,     # Quantum circuit function
            device=self.device,    # The device to run it on 
            interface="torch",     # Integrate with PyTorch
            diff_method="backprop",# Use backpropagation for gradients
        )

    def circuit(self, inp, labels=None, timesteps=None):
        """
        The quantum circuit for the PQC model. All the circuit variations consists of an
        AmplitudeEmbedding layer and StronglyEntanglingLayers. In case of label and timestep
        embeddings, RX gates are added.

        Args:
            inp (torch.Tensor): The input data.
            labels (torch.Tensor): The labels.
            timesteps (torch.Tensor): The timesteps.

        Returns:
            torch.Tensor: The output probabilities.
        """
        qml.AmplitudeEmbedding(
            features=inp,
            wires=range(self.dataset.num_wires()),
            normalize=True,
            pad_with=0.0,
        )

        if self.embeds.label_embedding() and labels is not None:
            qml.RX(phi=labels, wires=self.wires + self.embeds.label_wire())

        if self.embeds.timestep_embedding() and timesteps is not None:
            qml.RX(phi=timesteps, wires=self.wires + self.embeds.timestep_wire())

        qml.StronglyEntanglingLayers(
            weights=qw_map.tanh(self.weights),
            wires=range(self.wires),
        )

        return qml.probs(wires=range(self.wires))

    def forward(self, *args, **kwargs):
        """
        The forward pass of the PQC model.
        """
        x = self.qnode(*args, **kwargs)
        return x[:, : self.dataset.num_features()]
