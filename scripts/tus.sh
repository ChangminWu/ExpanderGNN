use_gpu=True
script=main_tu_graph_classification.py
datasets=("ENZYMES" "DD" "PROTEINS_full" "IMDB-BINARY" "REDDIT-BINARY")
models=('GCN' 'GIN' 'MLP')
savedir="results/tu-samplers/"
densities=(0.1 0.5 0.9)

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
      python $script --dataset "$i" --out_dir $savedir --experiment "random-expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "random"
      python $script --dataset "$i" --out_dir $savedir --experiment "rotate-expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "rotate"
    done
  done
done