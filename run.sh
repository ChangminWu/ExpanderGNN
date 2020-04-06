code=benchmark_graphclassification.py

dataset=ENZYMES
python $code --dataset $dataset --model GCN --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGCN --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model MLP --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderMLP --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model GIN --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGIN --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model GIN --expander_size 16 --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGIN --expander_size 16 --config 'configs/graph_classification_GIN_ENZYMES.json'

sleep 15

python $code --dataset $dataset --model GCN --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGCN --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model MLP --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderMLP --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model GIN --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGIN --config 'configs/graph_classification_GIN_ENZYMES.json'

sleep 15

python $code --dataset $dataset --model GCN --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGCN --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model MLP --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderMLP --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model GIN --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGIN --config 'configs/graph_classification_GIN_ENZYMES.json'

sleep 15

python $code --dataset $dataset --model GCN --expander_size 32 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGCN --expander_size 32 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model MLP --expander_size 32 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderMLP --expander_size 32 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model GIN --expander_size 24 --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGIN --expander_size 24 --config 'configs/graph_classification_GIN_ENZYMES.json'

sleep 15

python $code --dataset $dataset --model GCN --expander_size 48 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGCN --expander_size 48 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model MLP --expander_size 48 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderMLP --expander_size 48 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model GIN --expander_size 36 --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGIN --expander_size 36 --config 'configs/graph_classification_GIN_ENZYMES.json'

sleep 15

python $code --dataset $dataset --model GCN --expander_size 64 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGCN --expander_size 64 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model MLP --expander_size 64 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderMLP --expander_size 64 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model GIN --expander_size 48 --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGIN --expander_size 48 --config 'configs/graph_classification_GIN_ENZYMES.json'

sleep 15

python $code --dataset $dataset --model GCN --expander_size 80 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGCN --expander_size 80 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model MLP --expander_size 80 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderMLP --expander_size 80 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model GIN --expander_size 60 --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGIN --expander_size 60 --config 'configs/graph_classification_GIN_ENZYMES.json'

sleep 15

python $code --dataset $dataset --model GCN --expander_size 96 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGCN --expander_size 96 --config 'configs/graph_classification_GCN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model MLP --expander_size 96 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderMLP --expander_size 96 --config 'configs/graph_classification_MLP_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model GIN --expander_size 72 --config 'configs/graph_classification_GIN_ENZYMES.json'
sleep 10
python $code --dataset $dataset --model ExpanderGIN --expander_size 72 --config 'configs/graph_classification_GIN_ENZYMES.json'
#
#python $code --dataset $dataset --model ExpanderGCN --L 8 --config 'configs/graph_classification_GCN_ENZYMES.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderMLP --L 8 --config 'configs/graph_classification_MLP_ENZYMES.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderGIN --L 8 --config 'configs/graph_classification_GIN_ENZYMES.json'
#
#
#sleep 20
#
#dataset=DD
#python $code --dataset $dataset --model GCN --config 'configs/graph_classification_GCN_DD.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderGCN --config 'configs/graph_classification_GCN_DD.json'
#sleep 10
#python $code --dataset $dataset --model MLP --config 'configs/graph_classification_MLP_DD.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderMLP --config 'configs/graph_classification_MLP_DD.json'
#sleep 10
#python $code --dataset $dataset --model GIN --config 'configs/graph_classification_GIN_DD.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderGIN --config 'configs/graph_classification_GIN_DD.json'
#
#sleep 15
#
#python $code --dataset $dataset --model ExpanderGCN --L 8 --config 'configs/graph_classification_GCN_DD.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderMLP --L 8 --config 'configs/graph_classification_MLP_DD.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderGIN --L 8 --config 'configs/graph_classification_GIN_DD.json'
#
#sleep 20
#
#dataset=PROTEINS_full
#python $code --dataset $dataset --model GCN --config 'configs/graph_classification_GCN_PROTEINS_full.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderGCN --config 'configs/graph_classification_GCN_PROTEINS_full.json'
#sleep 10
#python $code --dataset $dataset --model MLP --config 'configs/graph_classification_MLP_PROTEINS_full.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderMLP --config 'configs/graph_classification_MLP_PROTEINS_full.json'
#sleep 10
#python $code --dataset $dataset --model GIN --config 'configs/graph_classification_GIN_PROTEINS_full.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderGIN --config 'configs/graph_classification_GIN_PROTEINS_full.json'
#
#sleep 15
#
#python $code --dataset $dataset --model ExpanderGCN --L 8 --config 'configs/graph_classification_GCN_PROTEINS_full.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderMLP --L 8 --config 'configs/graph_classification_MLP_PROTEINS_full.json'
#sleep 10
#python $code --dataset $dataset --model ExpanderGIN --L 8 --config 'configs/graph_classification_GIN_PROTEINS_full.json'

