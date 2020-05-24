code=main.py
datasets=("ENZYMES")
methods=('ExpanderGCN' 'ExpanderGatedGCN' 'ExpanderMLP' 'ExpanderGIN' 'ExpanderSimpleGCN' 'ExpanderGraphSage' 'GCN' 'GatedGCN' 'MLP' 'SimpleGCN' 'GIN' 'GraphSage')
sparsity=( 0.1 )
for i in "${datasets[@]}"
do
  for j in "${sparsity[@]}"
  do
    for k in "${methods[@]}"
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
	  echo "wrong method"
	  exit
      fi
      python $code --dataset ${i} --experiment "test" --model ${k} --sparsity ${j} --config ${config_file}
      sleep 15
     done
   done
done