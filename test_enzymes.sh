script=main_tu_graph_classification.py
datasets=("ENZYMES")
actives=('relu' 'prelu' 'linear' 'brelu' 'brelu-intercept' 'conv' 'rrelu' 'elu' 'sigmoid' 'tanh' 'lelu' 'softplus')
models=('GCN' 'GIN' 'MLP' 'GraphSage' 'GatedGCN' 'GAT' 'PNA')
densities=( 0.1 0.5 0.9 )
savedir="results/test/"

for i in "${datasets[@]}"
do
  for j in "${models[@]}"
  do
    config_file=configs/tu_graph_classification/${j}_${i}_100k.json
    for d in "${densities[@]}"
    do
      python $script --dataset ${i} --out_dir ${savedir} --experiment "expander-density-${d}" --model ${j} --density ${d} --linear_type "expander" --config ${config_file} --epochs 2 --mlp_layers 1 --use_gpu $use_gpu -num_split 3
    done
    for a in "${actives[@]}"
    do
      python $script --dataset ${i} --out_dir ${savedir} --experiment "activations-${a}" --model "Activation${j}" --activation ${a} --config ${config_file} --epochs 2 --mlp_layers 1 --use_gpu $use_gpu --num_split 3
    done
    python $script --dataset ${i} --out_dir ${savedir} --experiment "simple" --model "Simple${j}" --config ${config_file} --epochs 2 --mlp_layers 1 --use_gpu $use_gpu -num_split 3
    python $script --dataset ${i} --out_dir ${savedir} --experiment "regular" --model ${j} --density 1.0 --config ${config_file} --epochs 2 --mlp_layers 1 --use_gpu $use_gpu -num_split 3
  done
done
