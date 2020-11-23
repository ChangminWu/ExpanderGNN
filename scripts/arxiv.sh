use_gpu=True
models=('GCN')

#densities=(0.01 0.02 0.05 0.08 0.1 0.5 0.9)
#
#script=main_arxiv_node_classification.py
#savedir="results/arxiv-samplers/"
#datasets=('ogbn-arxiv')
#for j in "${models[@]}"
#do
#  for i in "${datasets[@]}"
#  do
#    config_file=configs/citation_node_classification/${j}_citation_100k.json
#    for d in "${densities[@]}"
#    do
#      python $script --dataset ogbn-arxiv --out_dir $savedir --experiment "random-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "random" --dropout 0.5 --epochs 1000 --init_lr 0.005
#      python $script --dataset ogbn-arxiv --out_dir $savedir --experiment "rotate-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "rotate" --dropout 0.5 --epochs 1000 --init_lr 0.005
#      python $script --dataset ogbn-arxiv --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "regular" --dropout 0.5 --epochs 1000 --init_lr 0.005
#    done
#  done
#done
#
#script=main_citation_node_classification.py
#savedir="results/citation-samplers/"
#datasets=("CORA" "CITESEER" "PUBMED")
#for j in "${models[@]}"
#do
#  for i in "${datasets[@]}"
#  do
#    config_file=configs/citation_node_classification/${j}_citation_100k.json
#    for d in "${densities[@]}"
#    do
#      python $script --dataset "$i" --out_dir $savedir --experiment "random-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "random" --batch_norm False
#      python $script --dataset "$i" --out_dir $savedir --experiment "rotate-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "rotate" --batch_norm False
#      python $script --dataset "$i" --out_dir $savedir --experiment "expander-density-${d}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --sampler "regular" --batch_norm False
#    done
#  done
#done

hiddims=(8 16 32 64 128 192)

#script=main_arxiv_node_classification.py
#savedir="results/arxiv-hiddims/"
#datasets=('ogbn-arxiv')
#for j in "${models[@]}"
#do
#  for i in "${datasets[@]}"
#  do
#    config_file=configs/citation_node_classification/${j}_citation_100k.json
#    for h in "${hiddims[@]}"
#    do
#      python $script --dataset ogbn-arxiv --out_dir $savedir --experiment "regular-dim-${h}" --model "$j" --density 1.0 --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --dropout 0.5 --epochs 1000 --init_lr 0.005
#    done
#  done
#done

script=main_citation_node_classification.py
savedir="results/citation-hiddims/"
datasets=("CORA" "CITESEER" "PUBMED")
for j in "${models[@]}"
do
  for i in "${datasets[@]}"
  do
    config_file=configs/citation_node_classification/${j}_citation_100k.json
    for h in "${hiddims[@]}"
    do
      python $script --dataset "$i" --out_dir $savedir --experiment "regular-dim-${h}" --model "$j" --density 1.0 --linear_type "regular" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --batch_norm False
    done
  done
done


densities=(0.05 0.1 0.5 0.9)

script=main_citation_node_classification.py
savedir="results/citation-small-models/"
datasets=("CORA" "CITESEER" "PUBMED")
for j in "${models[@]}"
do
  for i in "${datasets[@]}"
  do
    config_file=configs/citation_node_classification/${j}_citation_100k.json
    for d in "${densities[@]}"
    do
      python $script --dataset "$i" --out_dir $savedir --experiment "small-model-density-${h}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --batch_norm False
    done
  done
done

script=main_arxiv_node_classification.py
savedir="results/arxiv-small-models/"
datasets=('ogbn-arxiv')
for j in "${models[@]}"
do
  for i in "${datasets[@]}"
  do
    config_file=configs/citation_node_classification/${j}_citation_100k.json
    for d in "${densities[@]}"
    do
      python $script --dataset "$i" --out_dir $savedir --experiment "small-model-density-${h}" --model "$j" --density "$d" --linear_type "expander" --config "$config_file" --mlp_layers 1 --use_gpu $use_gpu --dropout 0.5 --epochs 1000 --init_lr 0.005
    done
  done
done