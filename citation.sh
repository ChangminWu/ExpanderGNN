use_gpu="True"
script=main_citation_node_classification.py
datasets=("CORA" "CITESEER" "PUBMED")
actives=('relu' 'prelu' 'brelu' 'rrelu' 'linear' 'softshrink' 'tanh' 'softplus' 'selu' 'lelu')
models=('PNA') #'GCN' 'GIN' 'MLP' 'GraphSage'
densities=( 0.1 0.5 0.9 )
savedir="results/citation-runs-new-gatedgcn/"

for i in "${datasets[@]}"
do
  for j in "${models[@]}"
  do
    config_file=configs/citation_node_classification/${j}_citation_100k.json
#    for d in "${densities[@]}"
#    do
#      python $script --dataset ${i} --out_dir ${savedir} --experiment "expander-density-${d}" --model ${j} --density ${d} --linear_type "expander" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --batch_norm False
#    done

    if [ "$j" != "MLP" ]
    then
#      for a in "${actives[@]}"
#      do
#        python $script --dataset ${i} --out_dir ${savedir} --experiment "activations-${a}" --model "Activation${j}" --activation ${a} --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --batch_norm False --init_lr 0.2
#      done
      python $script --dataset ${i} --out_dir ${savedir} --experiment "simple" --model "Simple${j}" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --batch_norm False --init_lr 0.2
    fi

#    python $script --dataset ${i} --out_dir ${savedir} --experiment "regular" --model ${j} --linear_type "regular" --density 1.0 --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --batch_norm False
  done
done
