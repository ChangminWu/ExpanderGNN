code=benchmark_graphclassification.py
datasets=("ENZYMES")
methods=("ExpanderMLP")
sizes=( 2 6 8 10 14 )
for i in "${datasets[@]}"
do
  for j in "${sizes[@]}"
  do
    for k in "${methods[@]}"
    do
      if [[ $k == *"GCN"* ]];
      then
          config_file=configs/graph_classification_GCN_${i}.json
      elif [[ $k == *"GIN"* ]];
      then
	  config_file=configs/graph_classification_GIN_${i}.json
      elif [[ $k == *"MLP"* ]];     
      then
	  config_file=configs/graph_classification_MLP_${i}.json
      else
	  echo "wrong method"
	  exit
      fi
      python $code --dataset ${i} --model ${k} --expander_size ${j} --config ${config_file}
      sleep 15
     done
   done
done
