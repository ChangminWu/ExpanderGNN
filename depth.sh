script=main.py
datasets=("ENZYMES")
models=('GCN' 'GIN' 'SimpleGCN')
expander_models=('ExpanderGCN' 'ExpanderGIN' 'ExpanderSimpleGCN' )
depth=(2 4 6 8 12 16)

for i in "${datasets[@]}"
do
  for k in "${expander_models[@]}"
  do
    if [[ $k == *"SimpleGCN"* ]];
    then
        config_file=configs/graph_classification_SimpleGCN_${i}.json
    elif [[ $k == *"GatedGCN"* ]];
    then
        config_file=configs/graph_classification_GatedGCN_${i}.json
    elif [[ $k == *"GCN"* ]];
    then
        config_file=configs/graph_classification_GCN_${i}.json
    elif [[ $k == *"GIN"* ]];
    then
        config_file=configs/graph_classification_GIN_${i}.json
    elif [[ $k == *"MLP"* ]];
    then
        config_file=configs/graph_classification_MLP_${i}.json
    elif [[ $k == *"Sage"* ]];
    then
        config_file=configs/graph_classification_GraphSage_${i}.json
    else
        echo "wrong model name"
        exit
    fi
    python $script --dataset ${i} --experiment "varing-depth-expander-l-2-mlp-1" --model ${k} --sparsity 0.1 --config ${config_file} --L 2 --n_mlp 1
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-4-mlp-1" --model ${k} --sparsity 0.1 --config ${config_file} --L 4 --n_mlp 1
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-2-mlp-2" --model ${k} --sparsity 0.1 --config ${config_file} --L 2 --n_mlp 2
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-6-mlp-1" --model ${k} --sparsity 0.1 --config ${config_file} --L 6 --n_mlp 1
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-3-mlp-2" --model ${k} --sparsity 0.1 --config ${config_file} --L 3 --n_mlp 2
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-2-mlp-3" --model ${k} --sparsity 0.1 --config ${config_file} --L 2 --n_mlp 3
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-8-mlp-1" --model ${k} --sparsity 0.1 --config ${config_file} --L 8 --n_mlp 1
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-4-mlp-2" --model ${k} --sparsity 0.1 --config ${config_file} --L 4 --n_mlp 2
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-2-mlp-4" --model ${k} --sparsity 0.1 --config ${config_file} --L 2 --n_mlp 4
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-12-mlp-1" --model ${k} --sparsity 0.1 --config ${config_file} --L 12 --n_mlp 1
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-6-mlp-2" --model ${k} --sparsity 0.1 --config ${config_file} --L 6 --n_mlp 2
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-4-mlp-3" --model ${k} --sparsity 0.1 --config ${config_file} --L 4 --n_mlp 3
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-3-mlp-4" --model ${k} --sparsity 0.1 --config ${config_file} --L 3 --n_mlp 4
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-16-mlp-1" --model ${k} --sparsity 0.1 --config ${config_file} --L 16 --n_mlp 1
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-8-mlp-2" --model ${k} --sparsity 0.1 --config ${config_file} --L 8 --n_mlp 2
    sleep 10
    python $script --dataset ${i} --experiment "varing-depth-expander-l-4-mlp-4" --model ${k} --sparsity 0.1 --config ${config_file} --L 4 --n_mlp 4
    sleep 10
  done
done

for i in "${datasets[@]}"
do
  for j in "${depth[@]}"
  do
    for k in "${models[@]}"
    do
      if [[ $k == *"SimpleGCN"* ]];
      then
          config_file=configs/graph_classification_SimpleGCN_${i}.json
      elif [[ $k == *"GatedGCN"* ]];
      then
          config_file=configs/graph_classification_GatedGCN_${i}.json
      elif [[ $k == *"GCN"* ]];
      then
          config_file=configs/graph_classification_GCN_${i}.json
      elif [[ $k == *"GIN"* ]];
      then
          config_file=configs/graph_classification_GIN_${i}.json
      elif [[ $k == *"MLP"* ]];
      then
          config_file=configs/graph_classification_MLP_${i}.json
      elif [[ $k == *"Sage"* ]];
      then
          config_file=configs/graph_classification_GraphSage_${i}.json
      else
          echo "wrong model name"
          exit
      fi
      python $script --dataset ${i} --experiment "varing-depth-normal-${j}" --model ${k} --sparsity 0.1 --config ${config_file} --L ${j}
      sleep 15
    done
  done
done