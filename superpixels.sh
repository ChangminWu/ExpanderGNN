use_gpu="False"
script=main_superpixel_graph_classification.py
datasets=("MNIST" "CIFAR10")
actives=('relu' 'prelu' 'linear' 'brelu' 'brelu-intercept' 'conv' 'rrelu' 'elu' 'sigmoid' 'tanh' 'lelu' 'softplus')
models=('GCN' 'GIN' 'MLP' 'GraphSage' 'GatedGCN' 'GAT' 'PNA')
densities=( 0.1 0.5 0.9 )
savedir="results/superpixel-runs/"

for i in "${datasets[@]}"
do
  for j in "${models[@]}"
  do
    config_file=configs/superpixel_graph_classification/${j}_${i}_100k.json
    for d in "${densities[@]}"
    do
      python $script --dataset ${i} --out_dir ${savedir} --experiment "expander-density-${d}" --model ${j} --density ${d} --linear_type "expander" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
    done
    for a in "${actives[@]}"
    do
      python $script --dataset ${i} --out_dir ${savedir} --experiment "activations-${a}" --model "Activation${j}" --activation ${a} --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
    done
    python $script --dataset ${i} --out_dir ${savedir} --experiment "simple" --model "Simple${j}" --config ${config_file} --mlp_layers 1
    python $script --dataset ${i} --out_dir ${savedir} --experiment "regular" --model ${j} --linear_type "regular" --density 1.0 --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
  done
done