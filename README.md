#  Sparsifying the Update Step in Graph Neural Networks
This repository is the official implementation of *Sparsifying the Update Step in Graph Neural Networks*.
![image](https://github.com/ChangminWu/ExpanderGNN/blob/public/img/illustration.jpg){:height="50%" width="50%"}

### Folder Structure
```
configs
  -- tu_graph_classification
  -- citation_node_classification
data
layers
  -- expander
  -- pna_utils
nets
scripts
train
```
Our implementation follows the pipeline proposed in [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982). We modify the codes from their [original repo](https://github.com/graphdeeplearning/benchmarking-gnns) to add expander and activation-only implementations in `layers` and `nets` folders. [Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718) (PNA) model was not initially included in [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982). We take the code from their [official implementation](https://github.com/lukecavabarrett/pna) and have made small modifications to adjust to [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982) pipeline. 

This code submission is able to reproduce experiments on four TU datasets (ENZYMES/DD/Proteins-full/IMDB-Binary integrated in [deep graph library](https://www.dgl.ai/)) and four Citation Datasets (Cora/Citeseer/Pubmed integrated in [deep graph library](https://www.dgl.ai/), ogbn-arxiv integrated in [open graph benchmark](https://ogb.stanford.edu/)).

Hyperparameters for model training of each dataset are stored in `configs` folder. We take hyperparameters settings from [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982) where in this benchmark paper they share the sets of hyperparameters that give their benchmark results. For citation datasets which are not included in [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982), we take hyperparameters settings that are shared by their official implementations in [SGC](https://arxiv.org/abs/1902.07153) and [open graph benchmark](https://ogb.stanford.edu/).

## Requirements
A virtual environment can be created by `conda` with the given environments file,
```
conda env create -f environments.yml
```

Notice that the [deep graph library](https://www.dgl.ai/) (dgl) is required be the latest release. It thus needs to be installed separately via
```
pip install --pre dgl-cu101
``` 

## Usage
To run batch of experiments shown in the paper, simply execute the bash file in `scripts`, e.g. to run experiments comparing expander model, activation-only model, SGC and vanilla model on TU datasets, do
```
bash scripts/tus.sh
```

For replicate the result of a specific model on a single dataset, for example, experiment of expander model with 10% density on arxiv dataset can be reproduced by

```
python main_arxiv_node_classification.py --dataset arxiv --out_dir ./example/ --experiment "expander-density-0.1" --model GCN --linear_type expander --density 0.1 --activations relu --config ./configs/citation_node_classification/GCN_citation_100k.json --mlp_layers 1 --use_gpu True
```

Different models can be switched by changing the input for parameter `linear_type` between `expander`/`regular` and for parameter `model` between `GCN`/`ActivationGCN`/`SimpleGCN`.

## Results
