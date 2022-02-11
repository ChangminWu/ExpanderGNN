script=graph_regression.py
densities=(0.95 1.0 0.05 0.2 0.3 0.4 0.6 0.7 0.8 0.9) #0.01 0.1 0.5 "prabhu" 
methods=("random")

savedir="results/zinc-sparse-1/"
for i in "${methods[@]}"
do
  for d in "${densities[@]}"
  do
    python $script --dataset zinc --outdir $savedir --num-readout-layers 1 --density "$d" --sample-method "$i" --weight-initializer glorot --use-expander 
  done

  python $script --dataset zinc --outdir $savedir --num-readout-layers 1 --density 1.0 --sample-method "$i" --weight-initializer glorot
done

savedir="results/zinc-sparse-3/"
for i in "${methods[@]}"
do  
  for d in "${densities[@]}"
  do
    python $script --dataset zinc --outdir $savedir --num-readout-layers 3 --density "$d" --sample-method "$i" --weight-initializer glorot --use-expander 
  done

  python $script --dataset zinc --outdir $savedir --num-readout-layers 3 --density 1.0 --sample-method "$i" --weight-initializer glorot

done

savedir="results/zinc-dense-1/"
for i in "${methods[@]}"
do
  python $script --dataset zinc --outdir $savedir --num-readout-layers 1 --density 1.0 --sample-method "$i" --weight-initializer glorot --dense-output
  
  for d in "${densities[@]}"
  do
    python $script --dataset zinc --outdir $savedir --num-readout-layers 1 --density "$d" --sample-method "$i" --weight-initializer glorot --use-expander --dense-output
  done

done

savedir="results/zinc-dense-3/"
for i in "${methods[@]}"
do
  python $script --dataset zinc --outdir $savedir --num-readout-layers 3 --density 1.0 --sample-method "$i" --weight-initializer glorot --dense-output
  
  for d in "${densities[@]}"
  do
    python $script --dataset zinc --outdir $savedir --num-readout-layers 3 --density "$d" --sample-method "$i" --weight-initializer glorot --use-expander --dense-output
  done

done