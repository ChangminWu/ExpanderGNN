use_gpu=True
script=main_molecule_graph_regression.py
datasets=("ZINC")
actives=('relu' 'prelu' 'brelu' 'rrelu' 'linear' 'softshrink' 'tanh' 'softplus' 'selu' 'lelu')
models=('GCN' 'GIN' 'MLP' 'GraphSage' 'GatedGCN' 'PNA')
densities=( 0.1 0.5 0.9 )
savedir="results/molecule-runs-new/"

for i in "${datasets[@]}"
do
  for j in "${models[@]}"
  do
    config_file=configs/molecule_graph_regression/${j}_${i}_100k.json
    for d in "${densities[@]}"
    do
      python $script --dataset ${i} --out_dir ${savedir} --experiment "expander-density-${d}" --model ${j} --density ${d} --linear_type "expander" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
    done

    if [ "$j" != "MLP" ]
    then
      for a in "${actives[@]}"
      do
        python $script --dataset ${i} --out_dir ${savedir} --experiment "activations-${a}" --model "Activation${j}" --activation ${a} --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu --init_lr 0.01
      done
      python $script --dataset ${i} --out_dir ${savedir} --experiment "simple" --model "Simple${j}" --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
    fi
    python $script --dataset ${i} --out_dir ${savedir} --experiment "regular" --model ${j} --linear_type "regular" --density 1.0 --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
  done
done

#use_gpu=True
#script=main_molecule_graph_regression.py
#datasets=("ZINC")
#actives=('relu' 'prelu' 'linear' 'brelu' 'brelu-intercept' 'conv' 'rrelu' 'elu' 'sigmoid' 'tanh' 'lelu' 'softplus')
#savedir="results/molecule-supplementary/"
#
#for i in "${datasets[@]}"
#do
#  config_file=configs/molecule_graph_regression/GraphSage_${i}_100k.json
#  python $script --dataset ${i} --out_dir ${savedir} --experiment "regular" --model "GraphSage" --linear_type "regular" --config ${config_file} --density 1.0 --mlp_layers 1 --use_gpu $use_gpu
#  for a in "${actives[@]}"
#  do
#    config_file=configs/molecule_graph_regression/GIN_${i}_100k.json
#    python $script --dataset ${i} --out_dir ${savedir} --experiment "activations-${a}" --model "ActivationGIN" --activation ${a} --config ${config_file} --mlp_layers 1 --use_gpu $use_gpu
#  done
#done
