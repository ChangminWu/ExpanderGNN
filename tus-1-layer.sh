use_gpu=True

script=main_citation_node_classification.py
datasets=("CORA" "CITESEER" "PUBMED")
actives=('relu' 'param')
savedir="results/citation-weights-one-layer/"
for i in "${datasets[@]}"
do
  config_file=configs/citation_node_classification/GCN_citation_100k.json
  for j in "${actives[@]}"
  do
    python $script --dataset "$i" --out_dir $savedir --experiment "${j}-random-normalize" --model GCN --linear_type "random-normalize" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --batch_norm False --init_lr 0.2
    python $script --dataset "$i" --out_dir $savedir --experiment "${j}-random" --model GCN --linear_type "random" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --batch_norm False --init_lr 0.2
    python $script --dataset "$i" --out_dir $savedir --experiment "${j}-regular" --model GCN --linear_type "regular" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --batch_norm False
    python $script --dataset "$i" --out_dir $savedir --experiment "${j}-activation" --model ActivationGCN --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --init_lr 0.2 --batch_norm False
  done
done


datasets=("ENZYMES" "DD" "PROTEINS_full" "IMDB-BINARY" "REDDIT-BINARY")
script=main_tu_graph_classification.py
actives=('relu' 'param')
savedir="results/tu-weights-one-layer/"
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
  python $script --dataset "$i" --out_dir $savedir --experiment "${j}-random-normalize" --model GCN --linear_type "random-normalize" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --L 1
  python $script --dataset "$i" --out_dir $savedir --experiment "${j}-random" --model GCN --linear_type "random" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --L 1
  python $script --dataset "$i" --out_dir $savedir --experiment "${j}-regular" --model GCN --linear_type "regular" --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --L 1
  python $script --dataset "$i" --out_dir $savedir --experiment "${j}-activation" --model ActivationGCN --config "$config_file" --use_gpu $use_gpu --mlp_layers 1 --activation "$j" --order 2 --L 1
  done
done

