from data.TUs import TUsDataset
from data.superpixels import SuperPixDataset
from data.molecules import MoleculeDataset


def LoadData(DATASET_NAME):
    TU_DATASETS = ["COLLAB", "ENZYMES", "DD", "PROTEINS_full", "IMDB-BINARY",
                   "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K",
                   "REDDIT-MULTI-12K", "PTC-MR"]
    if DATASET_NAME in TU_DATASETS:
        return TUsDataset(DATASET_NAME)

    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)

    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)
