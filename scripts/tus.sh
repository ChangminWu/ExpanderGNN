use_gpu=True
script=main_tu_graph_classification.py
datasets=("ENZYMES" "DD" "PROTEINS_full" "IMDB-BINARY")
actives=('relu' 'prelu' 'tanh')
models=('GCN' 'GIN' 'MLP')
densities=( 0.1 0.5 0.9 )
savedir="results/tu-results/"

for j in "${models[@]}"
do
  for i in "${datasets[@]}"
  do
    if [[ "$i" == "IMDB-BINARY" ]]
    then
      config_file=configs/tu_graph_classification/${j}_DD_100k.json
    else
      config_file=configs/tu_graph_classification/${j}_${i}_100k.json
    fi

    ############################### 
    ##### Expander Models
    ###############################
    
    for d in "${densities[@]}"
    do
      python $script --dataset "$i" --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu 
    done

    ############################### 
    ##### Activation-only Models
    ###############################    

    if [ "$j" != "MLP" ]
    then
      for a in "${actives[@]}"
      do
        python $script --dataset "$i" --out_dir $savedir --experiment "activations-${a}" --model "Activation${j}" --density 1.0 --linear_type "regular" --activation "$a" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu
      done
    fi

    ############################### 
    ##### SGC
    ###############################

    if [ "$j" = "GCN" ]
    then
      python $script --dataset "$i" --out_dir $savedir --experiment "simple" --model "Simple${j}" --density 1.0 --linear_type "regular" --config $config_file --mlp_layers 1 --use_gpu $use_gpu
    fi

    ############################### 
    ##### Vanilla Model
    ###############################
    
    python $script --dataset "$i" --out_dir $savedir --experiment "regular" --model "$j" --density 1.0 --linear_type "regular" --config $config_file --mlp_layers 1 --use_gpu $use_gpu
  done
done