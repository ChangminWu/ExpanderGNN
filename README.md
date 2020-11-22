# Analysing the Update step in Graph Neural Networks via Sparsification
This code sumbission gives examples to reproduce experiment results shown in our paper
*Analysing the Update step in Graph Neural Networks via Sparsification*.

We follow the pipeline proposed in [Dwivedi et al. 2020](https://arxiv.org/pdf/2003.00982.pdf). 

### Dependencies
```
pytorch 1.5.1
dgl-cu101 0.6
ogb 1.2.3
cuda 10.1
tensorboardX 2.1
numpy 1.18.1
scikit-learn 1.22.1
networkx 2.5
torch_scatter 2.0.4
tqdm 4.46.1
```
Notice that the [deep graph library](https://www.dgl.ai/) (dgl) needs to be the latest release. It can be installed via
```
pip install --pre dgl-cu101
```  
### Folder Structure


### 
```
python main_arxiv_node_classification.py --dataset arxiv --out_dir ./example/ --experiment "expander-density-0.1" --model GCN --linear_type expander --density 0.1 --config ./configs/citation_node_classification  --mlp_layers 1 --use_gpu True
```
