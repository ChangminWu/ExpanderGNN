from data.data_utils import TUsDataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for the TU Datasets
    TU_DATASETS = ['COLLAB', 'ENZYMES', 'DD', 'PROTEINS_full', 'IMDB-BINARY', 'IMDB-MULTI',
                   'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']
    if DATASET_NAME in TU_DATASETS: 
        return TUsDataset(DATASET_NAME)