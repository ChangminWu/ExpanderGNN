use_gpu=True
script=main_tu_graph_classification.py
datasets=("REDDIT-BINARY") # "ENZYMES" "DD" "PROTEINS_full" "IMDB-BINARY"
actives=('relu' 'prelu' 'linear' 'softshrink' 'tanh' 'selu' 'lelu') #'brelu' 'rrelu' 'softplus' 
models=('GCN' 'GIN' 'MLP') # 'GCN' 'PNA' 'GatedGCN' 'GraphSage' 'GIN' 'MLP'
densities=( 0.1 0.5 0.9 )
savedir="results/tu-runs-reddit/"

for j in "${models[@]}"
do
  for i in "${datasets[@]}"
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
