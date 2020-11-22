use_gpu=True
models=('GCN' 'GIN' 'MLP') #'GCN' 'GIN' 'MLP' 'GraphSage' 'PNA'
densities=( 0.1 0.5 0.9 )
actives=('tanh' 'relu' 'prelu')

savedir="results/node-classification-citations/"
script=main_citation_node_classification.py
datasets=("CORA" "CITESEER" "PUBMED")

for i in "${datasets[@]}"
do
  for j in "${models[@]}"
  do
    config_file=configs/citation_node_classification/${j}_citation_100k.json
    for d in "${densities[@]}"
    do
      if [ "$i" = "REDDIT" ]
      then
        python $script --dataset REDDIT --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu False --dropout 0.5 --epochs 50 --L 2
      else
        python $script --dataset "$i" --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --batch_norm False --L 2
      fi
   done

    if [ "$j" != "MLP" ]
    then
      for a in "${actives[@]}"
      do
      if [ "$i" = "REDDIT" ]
      then
        python $script --dataset REDDIT --out_dir $savedir --experiment "activations-${a}" --model "Activation${j}" --activation "$a" --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --init_lr 0.2 --batch_norm False --epochs 50 --L 2 --dropout 0.5
      else
        python $script --dataset "$i" --out_dir $savedir --experiment "activations-${a}" --model "Activation${j}" --activation "$a" --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --init_lr 0.2 --batch_norm False --L 2
      fi
      done
    fi

    if [ "$j" = "GCN" ]
    then
      if [ "$i" = "REDDIT" ]
      then
        python $script --dataset REDDIT --out_dir $savedir --experiment "simple" --model "Simple${j}" --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --init_lr 0.2 --batch_norm False --epochs 50 --L 2 --dropout 0.5
      else
        python $script --dataset "$i" --out_dir $savedir --experiment "simple" --model "Simple${j}" --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --init_lr 0.2 --batch_norm False --L 2
      fi
    fi

    if [ "$i" = "REDDIT" ]
    then
      python $script --dataset REDDIT --out_dir $savedir --experiment "regular" --model "$j" --linear_type "regular" --density 1.0 --config "$config_file" --mlp_layers 1 --use_gpu False --dropout 0.5 --epochs 50 --L 2
    else
      python $script --dataset "$i" --out_dir $savedir --experiment "regular" --model "$j" --linear_type "regular" --density 1.0 --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --batch_norm False --L 2
    fi
  done
done

savedir="results/node-classification-arxiv/"
script=main_arxiv_node_classification.py
for j in "${models[@]}"
do
  config_file=configs/citation_node_classification/${j}_citation_100k.json
  for d in "${densities[@]}"
  do
    python $script --dataset ogbn-arxiv --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --dropout 0.5 --epochs 1000 --init_lr 0.005 --L 2
  done

  if [ "$j" != "MLP" ]
  then
    for a in "${actives[@]}"
    do
      python $script --dataset ogbn-arxiv --out_dir $savedir --experiment "activations-${a}" --model "Activation${j}" --activation "$a" --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --dropout 0.5 --epochs 1000 --init_lr 0.1 --batch_norm False --L 2
    done
  fi

  if [ "$j" = "GCN" ]
  then
    python $script --dataset ogbn-arxiv --out_dir $savedir --experiment "simple" --model "Simple${j}" --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --dropout 0.5 --epochs 1000 --init_lr 0.1 --batch_norm False --L 2
  fi

  python $script --dataset ogbn-arxiv --out_dir $savedir --experiment "regular" --model "$j" --linear_type "regular" --density 1.0 --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --dropout 0.5 --epochs 1000 --init_lr 0.005 --L 2
done

