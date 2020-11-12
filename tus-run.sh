use_gpu=True
datasets=("ENZYMES" "DD" "PROTEINS_full" "IMDB-BINARY" "REDDIT-BINARY")


script=main_tu_graph_classification.py


actives=('relu' 'param')
savedir="results/tu-weights/"
for i in "${datasets[@]}"
do
  if [[ "$i" == "REDDIT-BINARY" || "$i" == "IMDB-BINARY" ]]
  then
    config_file=configs/tu_graph_classification/GCN_DD_100k.json
  else
    config_file=configs/tu_graph_classification/GCN_${i}_100k.json
  fi
  for j in "${actives[@]}"
  do
  python $script --dataset "$i" --out_dir $savedir --experiment "${j}-random-normalize" --model GCN --linear_type "random-normalize" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --hidden_dim 1000
  python $script --dataset "$i" --out_dir $savedir --experiment "${j}-random" --model GCN --linear_type "random" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --hidden_dim 1000
  python $script --dataset "$i" --out_dir $savedir --experiment "${j}-regular" --model GCN --linear_type "regular" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2
  python $script --dataset "$i" --out_dir $savedir --experiment "${j}-activation" --model ActivationGCN --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2
  done
done


orders=(2 3 4 5)
savedir="results/tu-orders/"
for i in "${datasets[@]}"
do
  if [[ "$i" == "REDDIT-BINARY" || "$i" == "IMDB-BINARY" ]]
  then
    config_file=configs/tu_graph_classification/GCN_DD_100k.json
  else
    config_file=configs/tu_graph_classification/GCN_${i}_100k.json
  fi

  for j in "${orders[@]}"
  do
    python $script --dataset "$i" --out_dir $savedir --experiment "param-order-${j}" --model GCN --linear_type "random-normalize" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "param" --order "$j"
  done
done


dims=(200 300 500 1000)
savedir="results/tu-dims/"
for i in "${datasets[@]}"
do
  if [[ "$i" == "REDDIT-BINARY" || "$i" == "IMDB-BINARY" ]]
  then
    config_file=configs/tu_graph_classification/GCN_DD_100k.json
  else
    config_file=configs/tu_graph_classification/GCN_${i}_100k.json
  fi

  for j in "${dims[@]}"
  do
    python $script --dataset "$i" --out_dir $savedir --experiment "param-dim-${j}" --model GCN --linear_type "random-normalize" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "param" --order 2 --hidden_dim "$j"
    python $script --dataset "$i" --out_dir $savedir --experiment "relu-dim-${j}" --model GCN --linear_type "regular" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "relu" --order 2 --hidden_dim "$j"
  done
done


layers=(2 6 8 10 15 20)
savedir="results/tu-layers/"
for i in "${datasets[@]}"
do
  if [[ "$i" == "REDDIT-BINARY" || "$i" == "IMDB-BINARY" ]]
  then
    config_file=configs/tu_graph_classification/GCN_DD_100k.json
  else
    config_file=configs/tu_graph_classification/GCN_${i}_100k.json
  fi

  for j in "${layers[@]}"
  do
    python $script --dataset "$i" --out_dir $savedir --experiment "param-layer-${j}" --model GCN --linear_type "random-normalize" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "param" --order 2 --L "$j"
    python $script --dataset "$i" --out_dir $savedir --experiment "relu-layer-${j}" --model GCN --linear_type "regular" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation relu --order 2 --L "$j"
  done
done