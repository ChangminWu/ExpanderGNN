from data.TUs import TUsDataset
from data.citations import CitationsDataset

def LoadData(DATASET_NAME):
    TU_DATASETS = ["COLLAB", "ENZYMES", "DD", "PROTEINS_full", "IMDB-BINARY",
                   "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K",
                   "REDDIT-MULTI-12K", "PTC-MR", "MUTAG"]
    if DATASET_NAME in TU_DATASETS:
        return TUsDataset(DATASET_NAME)

    CITATIONGRAPHS_DATASETS = ['CORA', 'CITESEER', 'PUBMED', 'CORA-FULL', 'MUTAG', 'COAUTHOR-CS', 'COAUTHOR-PHYSICS',
                               'REDDIT', 'AMAZON-PHOTO', 'AMAZON-COMPUTER']
    if DATASET_NAME in CITATIONGRAPHS_DATASETS:
        return CitationsDataset(DATASET_NAME)