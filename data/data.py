from data.TUs import TUsDataset
from data.superpixels import SuperPixDataset
from data.molecules import MoleculeDataset
from data.citations import CitationsDataset


def LoadData(DATASET_NAME):
    TU_DATASETS = ["COLLAB", "ENZYMES", "DD", "PROTEINS_full", "IMDB-BINARY",
                   "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K",
                   "REDDIT-MULTI-12K", "PTC-MR"]
    if DATASET_NAME in TU_DATASETS:
        return TUsDataset(DATASET_NAME)

    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)

    if DATASET_NAME == 'ZINC' or DATASET_NAME == 'ZINC-full':
        return MoleculeDataset(DATASET_NAME)

    CITATIONGRAPHS_DATASETS = ['CORA', 'CITESEER', 'PUBMED']
    if DATASET_NAME in CITATIONGRAPHS_DATASETS:
        return CitationsDataset(DATASET_NAME)
