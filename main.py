import argparse
import glob
import json
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.load_data import LoadData
from helpers import expand_writer, init_expander, get_model_param, weighted_expand_writer
from load_model import gnn_model
from train_graph_classification import train_epoch, evaluate_network


def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
    avg_test_acc = []
    avg_train_acc = []

    t0 = time.time()
    per_epoch_time = []
    per_epoch_memory = []

    dataset = LoadData(DATASET_NAME)
    
    if MODEL_NAME in ['SimpleGCN', 'ExpanderSimpleGCN']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for Simple GCN models (central node trick).")
            dataset._add_self_loops()
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file, write_expander_dir, write_weight_dir = dirs
    device = net_params['device']
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        saved_expander = OrderedDict()
        for split_number in range(10):
            saved_layers = dict()

            t0_split = time.time()
            log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
            writer = SummaryWriter(log_dir=log_dir)

            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device == 'cuda':
                torch.cuda.manual_seed(params['seed'])

            print("RUN NUMBER: ", split_number)
            trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], \
                                        dataset.test[split_number]

            print("Training Graphs: ", len(trainset))
            print("Validation Graphs: ", len(valset))
            print("Test Graphs: ", len(testset))
            print("Number of Classes: ", net_params['n_classes'])

            model = gnn_model(MODEL_NAME, net_params)

            saved_expander, _ = init_expander(model, saved_expander, saved_layers)
            model = model.to(device)
            if split_number == 0:
                expand_writer(saved_expander, curr_path=write_expander_dir)
                net_params['total_param'] = get_model_param(model, num = 0)
                print('MODEL/Total parameters:', MODEL_NAME, net_params['total_param'])

                # Write the network and optimization hyper-parameters in folder config/
                with open(write_config_file + '.txt', 'w') as f:
                    f.write(
                        """Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
                            DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'],
                                                             verbose=True)

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], [] 

            # batching exception for Diffpool
            drop_last = True if MODEL_NAME == 'DiffPool' else False

            train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
            val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
            test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

            with tqdm(range(params['epochs'])) as t:
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)    

                    start = time.time()
                    epoch_train_loss, epoch_train_acc, optimizer, writer = train_epoch(model, optimizer, device, train_loader, epoch, writer)

                    epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)

                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_accs.append(epoch_train_acc)
                    epoch_val_accs.append(epoch_val_acc)

                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                    writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                    writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                    _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)
                    t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                                  memory="{:.2%}".format(torch.cuda.max_memory_allocated(device=device)/total_memory),
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  test_acc=epoch_test_acc)  

                    per_epoch_time.append(time.time()-start)
                    per_epoch_memory.append(torch.cuda.max_memory_allocated(device=device))

                    # Saving checkpoint
                    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                    files = glob.glob(ckpt_dir + '/*.pkl')
                    for file in files:
                        epoch_nb = file.split('_')[-1]
                        epoch_nb = int(epoch_nb.split('.')[0])
                        if epoch_nb < epoch-1:
                            os.remove(file)

                    scheduler.step(epoch_val_loss)

                    if optimizer.param_groups[0]['lr'] < params['min_lr']:
                        print("\n!! LR EQUAL TO MIN LR SET.")
                        break
                        
                    # Stop training after params['max_time'] hours
                    if time.time()-t0_split > params['max_time']*3600/10:       # Dividing max_time by 10, since there are 10 runs in TUs
                        print('-' * 89)
                        print("Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(params['max_time']/10))
                        break

            _, test_acc = evaluate_network(model, device, test_loader, epoch)   
            _, train_acc = evaluate_network(model, device, train_loader, epoch)    
            avg_test_acc.append(test_acc)   
            avg_train_acc.append(train_acc)

            _ = weighted_expand_writer(model, saved_expander, saved_layers={},
                                       curr_path=write_weight_dir+"/split{}/".format(split_number))

            print("Test Accuracy [LAST EPOCH]: {:.4f}".format(test_acc))
            print("Train Accuracy [LAST EPOCH]: {:.4f}".format(train_acc))
            torch.cuda.reset_peak_memory_stats(device)
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')


    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time()-t0)/3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # Final test accuracy value averaged over 10-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}"""          .format(np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}"""          .format(np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)

    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n
    Total Time Taken: {:.4f} hrs\n Percentage of Average Memory taken per Epoch: {:.2%} \nAverage Time Per Epoch: {:.4f} s\n\n\nAll Splits Test Accuracies: {}"""
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100,
                  np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100,
               (time.time()-t0)/3600, np.mean(per_epoch_memory)/total_memory, np.mean(per_epoch_time), avg_test_acc))
        

#     # send results to gmail
#     try:
#         from gmail import send
#         subject = 'Result for Dataset: {}, Model: {}'.format(DATASET_NAME, MODEL_NAME)
#         body = """Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
#     FINAL RESULTS\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n
#     Total Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\nAll Splits Test Accuracies: {}"""\
#           .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
#                   np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100,
#                   np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100,
#                (time.time()-t0)/3600, np.mean(per_epoch_time), avg_test_acc)
#         send(subject, body)
#     except:
#         pass
        

def main():       
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--experiment', help="Please give a value for experiment name")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--sparsity', help="Please give a value for sparsity rate")
    parser.add_argument('--sparse_readout', help="Please give a value for sparsity readout")
    parser.add_argument('--mlp_readout', help="Please give a value for mlp readout")
    parser.add_argument('--activation', help="Please give a value for activation function")
    parser.add_argument('--neighbor_aggr_GCN', help="Please give a value for neighbor aggregation type")
    parser.add_argument('--neighbor_aggr_SGCN', help="Please give a value for neighbor aggregation type of SGCN")
    parser.add_argument('--n_mlp', help="Please give a value for number of layers in MLP")

    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    elif torch.cuda.is_available():
        config['gpu']['id'] = torch.cuda.device_count()
        config['gpu']['use'] = True
    else:
        config['gpu']['id'] = None
        config['gpu']['use'] = False
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # 
    
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']

    if args.experiment is not None:
        EXP_NAME = args.experiment
    else:
        EXP_NAME = config["experiment"]
    
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.sparsity is not None:
        net_params['sparsity'] = float(args.sparsity)
    if args.sparse_readout is not None:
        net_params['sparse_readout'] = True if args.sparse_readout == "True" else False
    if args.mlp_readout is not None:
        net_params['mlp_readout'] = True if args.mlp_readout == "True" else False
    if args.activation is not None:
        net_params['activation'] = "relu" if args.activation == "relu" else None
    if args.neighbor_aggr_GCN is not None:
        net_params['neighbor_aggr_GCN'] = args.neighbor_aggr_GCN
    if args.neighbor_aggr_SGCN is not None:
        net_params['neighbor_aggr_SGCN'] = args.neighbor_aggr_SGCN
    if args.n_mlp is not None:
        net_params['n_mlp'] = int(args.n_mlp)

    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = len(np.unique(dataset.all.graph_labels))
    net_params['n_classes'] = num_classes
    
    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
        max_num_node = max(num_nodes)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']
    
    root_log_dir = out_dir + 'logs/' + EXP_NAME + "_" + MODEL_NAME + "_" + DATASET_NAME  + "_sparsity_" + str(net_params["sparsity"])
    root_ckpt_dir = out_dir + 'checkpoints/' + EXP_NAME + "_" + MODEL_NAME + "_" + DATASET_NAME + "_sparsity_" + str(net_params["sparsity"])
    write_file_name = out_dir + 'results/result_' + EXP_NAME + "_" + MODEL_NAME + "_" + DATASET_NAME + "_sparsity_" + str(net_params["sparsity"])
    write_config_file = out_dir + 'configs/config_' + EXP_NAME + "_" + MODEL_NAME + "_" + DATASET_NAME + "_sparsity_" + str(net_params["sparsity"])
    write_expander_dir = out_dir + 'expanders/' +  EXP_NAME + "_" + MODEL_NAME + "_" + DATASET_NAME + "_sparsity_" + str(net_params["sparsity"])
    write_weight_dir = out_dir + 'weighted_expanders/' +  EXP_NAME + "_" + MODEL_NAME + "_" + DATASET_NAME + "_sparsity_" + str(net_params["sparsity"])
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file, write_expander_dir, write_weight_dir

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)


if __name__ == '__main__':
    main()    