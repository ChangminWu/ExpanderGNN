script=node_classification.py
densities=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0)
methods=("prabhu" "random")

savedir="results/arxiv-gcn-sparse/"
for i in "${methods[@]}"
do
  for d in "${densities[@]}"
  do
    python $script --dataset arxiv --outdir $savedir --num-layers 3 --hiddim 256 --density "$d" --sample-method "$i" --weight-initializer glorot --use-expander 
  done
  
  python $script --dataset arxiv --outdir $savedir --num-layers 3 --hiddim 256 --density "$d" --sample-method "$i" --weight-initializer glorot
done

savedir="results/arxiv-gcn-dense/"
for i in "${methods[@]}"
do
  for d in "${densities[@]}"
  do
    python $script --dataset arxiv --outdir $savedir --num-layers 3 --hiddim 256 --density "$d" --sample-method "$i" --weight-initializer glorot --dense-output --use-expander 
  done
  
  python $script --dataset arxiv --outdir $savedir --num-layers 3 --hiddim 256 --density "$d" --sample-method "$i" --weight-initializer glorot --dense-output
done

savedir="results/arxiv-sage-sparse/"
for i in "${methods[@]}"
do
  for d in "${densities[@]}"
  do
    python $script --dataset arxiv --outdir $savedir --num-layers 3 --hiddim 256 --density "$d" --sample-method "$i" --weight-initializer glorot --use-expander --use-sage 
  done
  
  python $script --dataset arxiv --outdir $savedir --num-layers 3 --hiddim 256 --density "$d" --sample-method "$i" --weight-initializer glorot --use-sage
done