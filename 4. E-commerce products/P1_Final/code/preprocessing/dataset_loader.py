from torch.utils.data import Dataset
import numpy as np

class PletsDataset(Dataset):
    """
        Dataset class for loading preprocessed data after the attribute extraction step.
        The dataset is loaded from a numpy file.
        Uses the pytorch dataset interface.
    """
    def __init__(self, file_paths: str, plets: str, direct: bool = False):
        """
            file_paths: The path to the directory containing the numpy file.
            plets: The name of the numpy file.
            direct: A boolean flag specifying whether to return include empty metadata in the returned entries.
        """
        self.file_paths = file_paths
        self.triplets=np.load(file_paths+f"/{plets}.npy")
        self.direct = direct

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        """
            Proxy function for getting an entry from the dataset.
            If direct is set to False, the function additionally returns empty metadata.
        """
        result = self.triplets[index].tolist()
        if self.direct:
            return result
        return result, []
