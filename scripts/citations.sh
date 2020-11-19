use_gpu=True
script=main_citation_node_classification.py
datasets=("CORA" "CITESEER" "PUBMED") #
actives=('relu' 'prelu' 'tanh') #
models=('GCN' 'GIN' 'MLP' 'GraphSage' 'PNA') #'GCN' 'GIN' 'MLP' 'GraphSage' 'PNA'
densities=( 0.1 0.5 0.9 )
savedir="results/citation-test/"

#for i in "${datasets[@]}"
#do
#  for j in "${models[@]}"
#  do
#    config_file=configs/citation_node_classification/${j}_citation_100k.json
#    for d in "${densities[@]}"
#    do
#      python $script --dataset "$i" --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --batch_norm False
#    done
#
#    if [ "$j" != "MLP" ]
#    then
#      for a in "${actives[@]}"
#      do
#        python $script --dataset "$i" --out_dir $savedir --experiment "activations-${a}" --model "Activation${j}" --activation "$a" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --init_lr 0.2 --batch_norm False
#      done
#    fi
#
#    if [ "$j" = "GCN"]
#    then
#      python $script --dataset "$i" --out_dir $savedir --experiment "simple" --model "Simple${j}" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --init_lr 0.2 --batch_norm False
#    fi
#
#    python $script --dataset "$i" --out_dir $savedir --experiment "regular" --model "$j" --linear_type "regular" --density 1.0 --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --batch_norm False
#  done
#done
#
#config_file=configs/citation_node_classification/GCN_citation_100k.json
#python $script --dataset REDDIT --out_dir $savedir --experiment "regular" --model "GCN" --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --batch_norm False
#python $script --dataset REDDIT --out_dir $savedir --experiment "activations-relu" --model "ActivationGCN" --activation relu --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --init_lr 0.2 --batch_norm False

script=main_arxiv_node_classification.py
config_file=configs/citation_node_classification/GCN_citation_100k.json
python $script --dataset ogbn-arxiv --out_dir $savedir --experiment "regular" --model "GCN" --epochs 500 --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --batch_norm False --in_feat_dropout 0.5 --dropout 0.5