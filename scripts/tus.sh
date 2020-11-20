#use_gpu=True
#script=main_tu_graph_classification.py
#datasets=("PROTEINS_full") # "IMDB-BINARY" "REDDIT-BINARY"
#models=('GCN' 'GIN' 'MLP')
#savedir="results/tu-hiddims/"
#densities=(0.01 0.02 0.05 0.08 0.1 0.5 0.9)

#for j in "${models[@]}"
#do
#  for i in "${datasets[@]}"
#  do
#    if [[ "$i" == "REDDIT-BINARY" || "$i" == "IMDB-BINARY" ]]
#    then
#      config_file=configs/tu_graph_classification/${j}_DD_100k.json
#    else
#      config_file=configs/tu_graph_classification/${j}_${i}_100k.json
#    fi
#
#    for d in "${densities[@]}"
#    do
#      python $script --dataset "$i" --out_dir $savedir --experiment "random-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "random"
#      python $script --dataset "$i" --out_dir $savedir --experiment "rotate-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "rotate"
#      python $script --dataset "$i" --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
#    done
#  done
#done

#hiddims=(5 10 30 50 70 90 110)
#for j in "${models[@]}"
#do
#  for i in "${datasets[@]}"
#  do
#    if [[ "$i" == "REDDIT-BINARY" || "$i" == "IMDB-BINARY" ]]
#    then
#      config_file=configs/tu_graph_classification/${j}_DD_100k.json
#    else
#      config_file=configs/tu_graph_classification/${j}_${i}_100k.json
#    fi
#
#    for h in "${hiddims[@]}"
#    do
#      python $script --dataset "$i" --out_dir $savedir --experiment "regular-dim-${h}" --model "$j" --density 1.0 --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "regular" --hidden_dim "$h" --out_dim "$h"
#    done
#  done
#done

use_gpu=True
script=main_tu_graph_classification.py
datasets=("MUTAG" "IMDB-MULTI" "REDDIT-BINARY") # "ENZYMES" "DD" "PROTEINS_full" "IMDB-BINARY"
actives=('relu' 'prelu' 'linear' 'softshrink' 'tanh' 'selu' 'lelu') #'brelu' 'rrelu' 'softplus'
models=('GCN' 'GIN' 'MLP') # 'GCN' 'PNA' 'GatedGCN' 'GraphSage' 'GIN' 'MLP'
densities=( 0.1 0.5 0.9 )
savedir="results/tu-newdata/"

for j in "${models[@]}"
do
  for i in "${datasets[@]}"
  do
    config_file=configs/tu_graph_classification/${j}_DD_100k.json
    for d in "${densities[@]}"
    do
      python $script --dataset "$i" --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
    done

    if [ "$j" != "MLP" ]
    then
      for a in "${actives[@]}"
      do
        python $script --dataset "$i" --out_dir $savedir --experiment "activations-${a}" --model "Activation${j}" --density 1.0 --linear_type "regular" --activation "$a" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
      done
    fi

    if [ "$j" = "GCN" ]
    then
      python $script --dataset "$i" --out_dir $savedir --experiment "simple" --model "Simple${j}" --density 1.0 --linear_type "regular" --config $config_file --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
    fi

    python $script --dataset "$i" --out_dir $savedir --experiment "regular" --model "$j" --density 1.0 --linear_type "regular" --config $config_file --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
  done
done