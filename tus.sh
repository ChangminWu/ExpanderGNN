use_gpu=True
script=main_tu_graph_classification.py
datasets=("ENZYMES" "DD" "PROTEINS_full" "IMDB-BINARY" "REDDIT-BINARY")
actives=('relu' 'prelu' 'brelu' 'rrelu' 'linear' 'softshrink' 'tanh' 'softplus' 'selu' 'lelu')
models=('GCN' 'GIN' 'MLP' 'GraphSage' 'GatedGCN' 'PNA')
densities=( 0.1 0.5 0.9 )
savedir="results/tu-runs-new/"

for i in "${datasets[@]}"
do
  for j in "${models[@]}"
  do
    if [[ "$i" == "REDDIT-BINARY" || "$i" == "IMDB-BINARY" ]]
    then
      config_file=configs/tu_graph_classification/${j}_DD_100k.json
    else
      config_file=configs/tu_graph_classification/${j}_${i}_100k.json
    fi
    for d in "${densities[@]}"
    do
      python $script --dataset ${i} --out_dir ${savedir} --experiment "expander-density-${d}" --model ${j} --density ${d} --linear_type "expander" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
    done

    if [ "$j" != "MLP" ]
    then
      for a in "${actives[@]}"
      do
        python $script --dataset ${i} --out_dir ${savedir} --experiment "activations-${a}" --model "Activation${j}" --activation ${a} --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
      done

      python $script --dataset ${i} --out_dir ${savedir} --experiment "simple" --model "Simple${j}" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
    fi

    python $script --dataset ${i} --out_dir ${savedir} --experiment "regular" --model ${j} --density 1.0 --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
  done
done
