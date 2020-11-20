use_gpu=True
script=main_molhiv_graph_classification.py
datasets=("ogbg-molhiv")
models=('GCN' 'GIN' 'MLP')
activs=('tanh' 'relu' 'prelu')
savedir="results/molhiv/"

for j in "${models[@]}"
do
  for i in "${datasets[@]}"
  do
    config_file=configs/tu_graph_classification/${j}_DD_100k.json
    for d in "${densities[@]}"
    do
      python $script --dataset "$i" --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --hidden_dim 300 --out_dim 300 --dropout 0.5 --init_lr 0.001
    done

    if [ "$j" != "MLP" ]
    then
      for a in "${actives[@]}"
      do
      python $script --dataset "$i" --out_dir $savedir --experiment "activations-${a}" --model "Activation${j}" --activation "$a" --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --hidden_dim 300 --out_dim 300 --dropout 0.5 --init_lr 0.001
      done
    fi

    if [ "$j" = "GCN" ]
    then
      python $script --dataset "$i" --out_dir $savedir --experiment "simple" --model "Simple${j}" --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --hidden_dim 300 --out_dim 300 --dropout 0.5 --init_lr 0.001
    fi

    python $script --dataset "$i" --out_dir $savedir --experiment "regular" --model "$j" --linear_type "regular" --density 1.0 --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --hidden_dim 300 --out_dim 300 --dropout 0.5 --init_lr 0.001
  done
done


#densities=(0.01 0.02 0.05 0.08 0.1 0.5 0.9)
#for j in "${models[@]}"
#do
#  for i in "${datasets[@]}"
#  do
#    config_file=configs/tu_graph_classification/${j}_DD_100k.json
#    for d in "${densities[@]}"
#    do
#      python $script --dataset "$i" --out_dir $savedir --experiment "random-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "random"
#      python $script --dataset "$i" --out_dir $savedir --experiment "rotate-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "rotate"
#      python $script --dataset "$i" --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "regular"
#    done
#  done
#done