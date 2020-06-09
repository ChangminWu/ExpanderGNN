script=main.py
datasets=("ENZYMES" "DD" "PROTEINS_full")
expandmodels=('ExpanderGCN' 'ExpanderMLP' 'ExpanderGIN' 'ExpanderGraphSage' 'ExpanderGatedGCN' 'ExpanderSimpleGCN' 'ExpanderSimpleGIN' 'ExpanderSimpleMLP' 'ExpanderSimpleGraphSage')
models=("SimpleGCN" "GCN" "SimpleGIN" "GIN" "SimpleMLP" "MLP" "SimpleGraphSage" "GraphSage" "GatedGCN")
sparsities=( 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 )
savename="results/10-06/"

for i in "${datasets[@]}"
do
  config_file=configs/graph_classification_SimpleGCN_${i}.json
  python $script --dataset ${i} --out_dir ${savename} --experiment "zero-sparsity-sgcn" --model "SimpleGCN" --sparsity 0.0 --config ${config_file} --n_mlp 1
  sleep 10
  config_file=configs/graph_classification_SimpleGIN_${i}.json
  python $script --dataset ${i} --out_dir ${savename} --experiment "zero-sparsity-sgin" --model "SimpleGIN" --sparsity 0.0 --config ${config_file} --n_mlp 1
  sleep 10
  config_file=configs/graph_classification_SimpleMLP_${i}.json
  python $script --dataset ${i} --out_dir ${savename} --experiment "zero-sparsity-smlp" --model "SimpleMLP" --sparsity 0.0 --config ${config_file} --n_mlp 1
  sleep 10
  config_file=configs/graph_classification_SimpleGraphSage_${i}.json
  python $script --dataset ${i} --out_dir ${savename} --experiment "zero-sparsity-sgraphsage" --model "SimpleGraphSage" --sparsity 0.0 --config ${config_file} --n_mlp 1
  sleep 10

  for j in "${sparsities[@]}"
  do
    for k in "${expandmodels[@]}"
    do
      if [[ $k == *"SimpleGCN"* ]];
      then
          config_file=configs/graph_classification_SimpleGCN_${i}.json
      elif [[ $k == *"SimpleGIN"* ]];
      then
          config_file=configs/graph_classification_SimpleGIN_${i}.json
      elif [[ $k == *"SimpleMLP"* ]];
      then
          config_file=configs/graph_classification_SimpleMLP_${i}.json
      elif [[ $k == *"SimpleGraphSage"* ]];
      then
          config_file=configs/graph_classification_SimpleGraphSage_${i}.json
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
      python $script --dataset ${i} --out_dir ${savename} --experiment "varing-sparsity-dense-readout" --model ${k} --sparsity ${j} --config ${config_file} --n_mlp 1 --sparse_readout False
      sleep 10
    done
  done

  sleep 30

  for j in "${sparsities[@]}"
  do
    for k in "${models[@]}"
    do
      if [[ $k == *"SimpleGCN"* ]];
      then
          config_file=configs/graph_classification_SimpleGCN_${i}.json
      elif [[ $k == *"SimpleGIN"* ]];
      then
          config_file=configs/graph_classification_SimpleGIN_${i}.json
      elif [[ $k == *"SimpleMLP"* ]];
      then
          config_file=configs/graph_classification_SimpleMLP_${i}.json
      elif [[ $k == *"SimpleGraphSage"* ]];
      then
          config_file=configs/graph_classification_SimpleGraphSage_${i}.json
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
      python $script --dataset ${i} --out_dir ${savename} --experiment "varing-sparsity-normal-models" --model ${k} --sparsity ${j} --config ${config_file} --n_mlp 1 --sparse_readout False
      sleep 15
    done
  done
done
