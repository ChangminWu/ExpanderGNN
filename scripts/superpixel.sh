use_gpu="True"
script=main_superpixels_graph_classification.py
datasets=("MNIST" "CIFAR10")
actives=('relu' 'prelu' 'tanh')
models=('GCN' 'GIN' 'GraphSage' 'MLP') #'PNA'
densities=( 0.1 0.5 0.9 )
savedir="results/superpixel-runs/"

for i in "${datasets[@]}"
do
  for j in "${models[@]}"
  do
    config_file=configs/superpixel_graph_classification/${j}_${i}_100k.json
    for d in "${densities[@]}"
    do
      python $script --dataset ${i} --out_dir ${savedir} --experiment "expander-density-${d}" --model ${j} --density ${d} --linear_type "expander" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
    done

    if [ "$j" != "MLP" ]
    then
      for a in "${actives[@]}"
      do
        python $script --dataset ${i} --out_dir ${savedir} --experiment "activations-${a}" --model "Activation${j}" --activation ${a} --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
      done

    if [ "$j" = "GCN" ]
    then
      python $script --dataset ${i} --out_dir ${savedir} --experiment "simple" --model "Simple${j}" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
    fi

    python $script --dataset ${i} --out_dir ${savedir} --experiment "regular" --model ${j} --linear_type "regular" --density 1.0 --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
  done
done

actives=('relu' 'prelu' 'tanh')
models=('PNA') #'PNA'
densities=( 0.1 0.5 0.9 )
savedir="results/superpixel-runs/"

for i in "${datasets[@]}"
do
  for j in "${models[@]}"
  do
    config_file=configs/superpixel_graph_classification/${j}_${i}_100k.json
    for d in "${densities[@]}"
    do
      python $script --dataset ${i} --out_dir ${savedir} --experiment "expander-density-${d}" --model ${j} --density ${d} --linear_type "expander" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
    done

    if [ "$j" != "MLP" ]
    then
      for a in "${actives[@]}"
      do
        python $script --dataset ${i} --out_dir ${savedir} --experiment "activations-${a}" --model "Activation${j}" --activation ${a} --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
      done

    if [ "$j" = "GCN" ]
    then
      python $script --dataset ${i} --out_dir ${savedir} --experiment "simple" --model "Simple${j}" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
    fi

    python $script --dataset ${i} --out_dir ${savedir} --experiment "regular" --model ${j} --linear_type "regular" --density 1.0 --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
  done
done