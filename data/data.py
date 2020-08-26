from data.tu_graph_classification import TUsDataset


def LoadData(DATASET_NAME):
    TU_DATASETS = ["COLLAB", "ENZYMES", "DD", "PROTEINS_full", "IMDB-BINARY",
                   "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K",
                   "REDDIT-MULTI-12K", "PTC-MR"]
    if DATASET_NAME in TU_DATASETS:
        return TUsDataset(DATASET_NAME)
