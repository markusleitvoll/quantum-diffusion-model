from dataclasses import dataclass
import torch
from utils.normalization import normalize_embeddings


@dataclass
class Embeddings:
    embeddings: list

    def num_ancilla(self):
        return len(self.embeddings)

    def timestep_embedding(self):
        return "timestep" in self.embeddings

    def label_embedding(self):
        return "label" in self.embeddings

    def both_embeddings(self):
        return self.timestep_embedding() and self.label_embedding()
    
    def get_folder_name(self):
        if self.both_embeddings(): return "combined"
        if self.timestep_embedding(): return "timestep"
        if self.label_embedding(): return "label"
        return "basic"

    def label_wire(self):
        if self.both_embeddings():
            return -2
        elif self.label_embedding():
            return -1
        return None

    def timestep_wire(self):
        if self.both_embeddings():
            return -1
        elif self.timestep_embedding():
            return -1
        return None

    @staticmethod
    def generate_labels(labels, total_count):
        repeated_labels = []
        full_repeats = total_count // len(labels)  # Full repeats of the label list

        # Extend each label 'full_repeats' times
        for label in labels:
            repeated_labels.extend([label] * full_repeats)

        # Append extra elements from the end if there's a remainder
        remainder = total_count % len(labels)
        repeated_labels.extend(labels[-remainder:] if remainder else [])

        # Convert to a float tensor and normalize
        label_tensor = torch.tensor(repeated_labels, dtype=torch.float)
        label_tensor = normalize_embeddings(label_tensor)

        return label_tensor
    

    @staticmethod
    def generate_timesteps(T):
        timesteps = torch.arange(T, dtype=torch.float)
        timesteps = normalize_embeddings(timesteps)
        
        return timesteps