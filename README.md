#  Sparsifying the Update Step in Graph Neural Networks
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository is the official implementation of [*Sparsifying the Update Step in Graph Neural Networks*](https://arxiv.org/abs/2109.00909).

<div align=center>
<img src=https://github.com/ChangminWu/ExpanderGNN/blob/public/img/illustration.jpg  width="50%">
</div>

Our implementation follows the pipeline proposed in [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982). We modify the code from their [original repo](https://github.com/graphdeeplearning/benchmarking-gnns) to add expander and activation-only implementations in the `layers` and `nets` folders. Note that for the [Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718) (PNA) model, which was not initially included in [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982), we have included the code from the [official implementation](https://github.com/lukecavabarrett/pna) with small modifications to adjust to the [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982) pipeline. 

## Requirements
An appropriate virtual environment can be created by `conda`  using the provided environments file,
```
conda env create -f environments.yml
```

Notice that the [deep graph library](https://www.dgl.ai/) (dgl) is required to be the latest release(>=0.6.0). It thus needs to be installed separately via
```
pip install dgl-cu101
``` 

## Usage
To run the full batch of experiments shown in the paper, simply execute the bash file in `scripts`, e.g. to run the experiments comparing the expander model, activation-only model, SGC and vanilla model on the TU datasets, execute
```
bash scripts/tus.sh
```

To replicate the result of a specific model on a single dataset, for example, the experiment of the expander model with 10% density on the arxiv dataset can be reproduced by

```
python main_arxiv_node_classification.py --dataset arxiv --out_dir ./example/ --experiment "expander-density-0.1" --model GCN --linear_type expander --density 0.1 --activations relu --config ./configs/citation_node_classification/GCN_citation_100k.json --mlp_layers 1 --use_gpu True
```

Different model types can be experimented with by choosing the input to the parameter `linear_type` from `expander`/`regular` and to the parameter `model` from `GCN`/`ActivationGCN`/`SimpleGCN`.

This repository contains code for reproducing experiments on four TU datasets (ENZYMES/DD/Proteins-full/IMDB-Binary integrated in [deep graph library](https://www.dgl.ai/)) and four Citation Datasets (Cora/Citeseer/Pubmed integrated in [deep graph library](https://www.dgl.ai/), ogbn-arxiv integrated in [open graph benchmark](https://ogb.stanford.edu/)).

Hyperparameters for model training of each dataset are stored in the `configs` folder. We take hyperparameters settings from the benchmark paper [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982), where they share the sets of hyperparameters that give their benchmark results. For the citation datasets which are not included in [Dwivedi et al. 2020](https://arxiv.org/abs/2003.00982), we take hyperparameters settings that are shared by their official implementations in [SGC](https://arxiv.org/abs/1902.07153) and [open graph benchmark](https://ogb.stanford.edu/).

## Results

>Accuracy for Graph (ENZYMES/Proteins) and Node (ogbn-arxiv) Classification 

| Methods\Datasets | ENZYMES | Proteins | ogbn-arxiv |
|------------------|---------|----------|------------|
| Vanilla GCN      | 66.50±8.71 | **76.73±3.85** | 71.22±0.76 |
| Expander GCN (50%)| 64.83±8.64 | 76.36±3.43 | **71.42±0.55** |
| Activation-only GCN (prelu) | **66.67±6.71** | 75.29±4.85 | 68.29±0.13 |
| SGC | 63.67±8.06 | 67.65±2.21 | 66.53±0.07 |

## Contribution
#### Authors: 
+ Johannes F. Lutzeyer*
+ Changmin Wu*
+ Michalis Vazirgiannis

*: Equal Contribution

The paper is accepted at ICLR 2022 workshop on *Geometrical and Topological Representation Learning.* If you find our repo useful, please cite
```
@misc{lutzeyer2021sparsifying,
      title={Sparsifying the Update Step in Graph Neural Networks}, 
      author={Johannes F. Lutzeyer and Changmin Wu and Michalis Vazirgiannis},
      year={2021},
      eprint={2109.00909},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

