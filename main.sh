script=main.py
datasets=("ENZYMES" "DD" "PROTEINS_full")
models=('ExpanderGCN' 'GCN' 'ExpanderMLP' 'MLP' 'ExpanderGIN' 'GIN' 'ExpanderGraphSage' 'GraphSage' 'ExpanderGatedGCN' 'GatedGCN' 'ExpanderSimpleGCN' 'SimpleGCN' )
sparsities=( 0.0 0.05 0.1 0.3 0.5 0.7 0.9 1.0 )

for i in "${datasets[@]}"
do
  for j in "${sparsities[@]}"
  do
    if [[ $j == 0.0 ]];
    then
      config_file=configs/graph_classification_SimpleGCN_${i}.json
      python $code --dataset ${i} --experiment "zero-sparsity-sgcn" --model "SimpleGCN" --sparsities ${j} --config ${config_file} --n_mlp 1
      sleep 15
    else
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
        python $code --dataset ${i} --experiment "varing-sparsity" --model ${k} --sparsities ${j} --config ${config_file} --n_mlp 1
        sleep 15
      done
    fi
  done
done
