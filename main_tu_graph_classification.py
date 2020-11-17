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
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.data import LoadData
from utils import expander_writer, expander_weights_writer, get_model_param, init_expander
from nets.tu_graph_classification.load_net import gnn_model
from train.train_tu_graph_classification import train_epoch, evaluate_network


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print("cuda available with GPU:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
    avg_test_acc = []
    avg_train_acc = []
    avg_convergence_epochs = []

    t0 = time.time()
    per_epoch_time = []
    per_epoch_memory = []
    per_split_train_inference_time = []
    per_split_test_inference_time = []

    dataset = LoadData(DATASET_NAME)

    if "GCN" in MODEL_NAME:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for Simple GCN models (central node trick).")
            dataset._add_self_loops()

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file, write_expander_dir, write_weight_dir = dirs
    device = net_params['device']

    if device.type == "cpu":
        total_memory = 1.
    elif device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        saved_expander = OrderedDict()
        for split_number in range(params["num_split"]):
            saved_layers = dict()

            t0_split = time.time()
            log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
            writer = SummaryWriter(log_dir=log_dir)

            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])

            print("RUN NUMBER: ", split_number)
            trainset, valset, testset = (dataset.train[split_number], dataset.val[split_number],
                                         dataset.test[split_number])

            print("Training Graphs: ", len(trainset))
            print("Validation Graphs: ", len(valset))
            print("Test Graphs: ", len(testset))
            print("Number of Classes: ", net_params['n_classes'])

            if 'PNA' in MODEL_NAME:
                D = torch.cat([
                    torch.sparse.sum(g.adjacency_matrix(transpose=True),
                                     dim=-1).to_dense()
                    for g in dataset.train[split_number].graph_lists])
                net_params['avg_d'] = dict(lin=torch.mean(D),
                                           exp=torch.mean(torch.exp(
                                               torch.div(1, D)) - 1),
                                           log=torch.mean(torch.log(D + 1)))

            model = gnn_model(MODEL_NAME, net_params)
            saved_expander, _ = init_expander(model, saved_expander,
                                              saved_layers)
            model = model.to(device)

            if split_number == 0:
                expander_writer(saved_expander, curr_path=write_expander_dir)
                net_params['total_param'] = get_model_param(model, num=0)
                print("MODEL/Total parameters:", MODEL_NAME, net_params["total_param"])

                # Write the network and optimization
                # hyper-parameters in folder config/
                with open(write_config_file + '.txt', 'w') as f:
                    f.write("""Dataset: {}\n
                               Model: {}\n
                               params={}\n
                               net_params={}\n
                               Total Parameters:{}\n\n""".format(DATASET_NAME, MODEL_NAME, params, net_params,
                                                                 net_params["total_param"]))

            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=params["lr_reduce_factor"],
                                                             patience=params["lr_schedule_patience"],
                                                             verbose=True)

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], []

            # batching exception for Diffpool
            drop_last = True if MODEL_NAME == 'DiffPool' else False

            train_loader = DataLoader(trainset,
                                      batch_size=params['batch_size'],
                                      shuffle=True,
                                      drop_last=drop_last,
                                      collate_fn=dataset.collate)
            val_loader = DataLoader(valset,
                                    batch_size=params['batch_size'],
                                    shuffle=False,
                                    drop_last=drop_last,
                                    collate_fn=dataset.collate)
            test_loader = DataLoader(testset,
                                     batch_size=params['batch_size'],
                                     shuffle=False,
                                     drop_last=drop_last,
                                     collate_fn=dataset.collate)

            ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(model.state_dict(), "{}.pkl".format(ckpt_dir + "/epoch_{}".format(0)))

            with tqdm(range(params['epochs'])) as t:
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)

                    start = time.time()
                    epoch_train_loss, epoch_train_acc, optimizer, writer = train_epoch(model, optimizer,
                                                                                       device, train_loader,
                                                                                       epoch, writer)

                    if device.type == "cuda":
                        per_epoch_memory.append(torch.cuda.max_memory_reserved(device=device))
                    elif device.type == "cpu":
                        per_epoch_memory.append(1.)

                    epoch_val_loss, epoch_val_acc =\
                        evaluate_network(model, device, val_loader)
                    _, epoch_test_acc =\
                        evaluate_network(model, device, test_loader)

                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_accs.append(epoch_train_acc)
                    epoch_val_accs.append(epoch_val_acc)

                    writer.add_scalar("train/_loss", epoch_train_loss, epoch)
                    writer.add_scalar("val/_loss", epoch_val_loss, epoch)
                    writer.add_scalar("train/_acc", epoch_train_acc, epoch)
                    writer.add_scalar("val/_acc", epoch_val_acc, epoch)
                    writer.add_scalar("test/_acc", epoch_test_acc, epoch)
                    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

                    t.set_postfix(time=time.time()-start,
                                  lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss,
                                  val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc,
                                  val_acc=epoch_val_acc,
                                  test_acc=epoch_test_acc)

                    per_epoch_time.append(time.time()-start)

                    # Saving checkpoint
                    torch.save(model.state_dict(), "{}.pkl".format(ckpt_dir + "/epoch_" + str(epoch+1)))

                    files = glob.glob(ckpt_dir + '/*.pkl')
                    for file in files:
                        epoch_nb = file.split('_')[-1]
                        epoch_nb = int(epoch_nb.split('.')[0])
                        if epoch_nb < epoch-1 and epoch_nb % 50 != 0:
                            os.remove(file)

                    scheduler.step(epoch_val_loss)

                    if optimizer.param_groups[0]["lr"] < params["min_lr"]:
                        print("\n!! LR EQUAL TO MIN LR SET.")
                        break

                    # Stop training after params["max_time"] hours
                    # Dividing max_time by 10, since there are 10 runs in TUs
                    if time.time()-t0_split > params['max_time']*3600/params["num_split"]:
                        print("-" * 89)
                        print("Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping"
                              .format(params['max_time']/params["num_split"]))
                        break

            start_inference_test = time.time()
            _, test_acc = evaluate_network(model, device, test_loader)
            per_split_test_inference_time.\
                append(time.time() - start_inference_test)
            start_inference_train = time.time()
            _, train_acc = evaluate_network(model, device, train_loader)
            per_split_train_inference_time.\
                append(time.time() - start_inference_train)
            avg_test_acc.append(test_acc)
            avg_train_acc.append(train_acc)
            avg_convergence_epochs.append(epoch)

            _ = expander_weights_writer(model, saved_expander, saved_layers={},
                                        curr_path=write_weight_dir+"/RUN_{}/".format(split_number))

            print("Test Accuracy [LAST EPOCH]: {:.4f}".format(test_acc))
            print("Train Accuracy [LAST EPOCH]: {:.4f}".format(train_acc))
            print("Convergence Time (Epochs): {:.4f}".format(epoch))

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early because of KeyboardInterrupt")

    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time()-t0)/3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    print("AVG CONVERGENCE Time (Epochs): {:.4f}".format(np.mean(np.array(avg_convergence_epochs))))
    print("AVG MEMORY PER EPOCH: {:.4%}".format(np.mean(per_epoch_memory)/total_memory))

    # Final test accuracy value averaged over K-fold
    print("""\n\nFINAL RESULTS\n\nTEST ACCURACY """
          """averaged: {:.4f} with s.d. {:.4f}""".format(np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY """
          """averaged: {:.4f} with s.d. {:.4f}""".format(np.mean(np.array(avg_train_acc))*100,
                                                         np.std(avg_train_acc)*100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)

    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {}\n\n Model: {}\n\n"""
                """params={}\n\n net_params={}\n\n Architecture: {}\n\n"""
                """Total Parameters: {}\n\n"""
                """FINAL RESULTS\n\n"""
                """TEST ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n"""
                """TRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n"""
                """Average Convergence Time (Epochs): {:.4f} with s.d. {:.4f}\n\n"""
                """Total Time Taken: {:.4f} hrs\n\n"""
                """Percentage of Average Memory taken per Epoch: {:.4%}\n\n"""
                """Average Time Per Epoch: {:.4f} s\n\n"""
                """Average Inference Time For Train Per Split: {:.4f} s\n\n"""
                """Average Inference Time For Test Per Split: {:.4f} s\n\n"""
                """All Splits Test Accuracies: {}"""
                .format(DATASET_NAME, MODEL_NAME, params,
                        net_params, model, net_params['total_param'],
                        np.mean(np.array(avg_test_acc))*100,
                        np.std(avg_test_acc)*100,
                        np.mean(np.array(avg_train_acc))*100,
                        np.std(avg_train_acc)*100,
                        np.mean(avg_convergence_epochs),
                        np.std(avg_convergence_epochs),
                        (time.time()-t0)/3600,
                        np.mean(per_epoch_memory)/total_memory,
                        np.mean(per_epoch_time),
                        np.mean(per_split_train_inference_time),
                        np.mean(per_split_test_inference_time),
                        avg_test_acc))


def main():
    parser = argparse.ArgumentParser()

    """
    general parameters
    """
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--use_gpu', default="true", help="Please give a value for using gpu or not")
    parser.add_argument('--experiment', help="Please give a value for experiment name")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--num_split', type=int, help="Please give a value for split numbers")

    """
    Training parameters
    """
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--max_time', help="Please give a value for max_time")

    """
    Model parameters
    """
    parser.add_argument('--L', help="Please give a value for number of layers")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden dimension")
    parser.add_argument('--out_dim', help="Please give a value for output dimension")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--graph_pool', help="Please give a value for graph_pool")
    parser.add_argument('--neighbor_pool', help="Please give a value for neighbor aggregation type")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--activation', help="Please give a value for activation function")
    parser.add_argument('--mlp_layers', help="Please give a value for number of layers in MLP")
    parser.add_argument('--bias', help="Please give a value for bias")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--dgl_builtin', help="Please give a value for whether using dgl_builtin")

    """
    Expander parameters
    """
    parser.add_argument('--density', help="Please give a value for Expander density")
    parser.add_argument('--linear_type', help="Please give a value for linear layer type")
    parser.add_argument('--sampler', help="Please give a value for expander samplers")

    """
    Special parameters for MLP net
    """
    parser.add_argument('--gated', help="Please give a value for gated")

    """
    Special parameters for PNA
    """
    parser.add_argument('--aggregators', type=str, help="Aggregators to use.")
    parser.add_argument('--scalers', type=str, help="Scalers to use.")
    parser.add_argument('--num_tower', type=int, help="number of towers to use.")
    parser.add_argument('--divide_input', help="Whether to divide the input.")
    parser.add_argument('--gru', help="Whether to use gru.")
    parser.add_argument('--edge_dim', type=int, help="Size of edge embeddings.")
    parser.add_argument('--num_pretrans_layer', type=int, help="number of pretrans layers.")
    parser.add_argument('--num_posttrans_layer', type=int, help="number of posttrans layers.")
    parser.add_argument('--use_simplified_version', help="whether to use simplified PNA.")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config["gpu"]["id"] = int(args.gpu_id)
    elif torch.cuda.is_available():
        config["gpu"]["id"] = torch.cuda.device_count()-1
    else:
        config["gpu"]["id"] = None
    config["gpu"]["use"] = True if args.use_gpu == "True" else False
    if config["gpu"]["id"] is not None and config["gpu"]["use"]:
        config["gpu"]["use"] = True
        print("cuda available with GPU:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        config["gpu"]["use"] = False
        print("cuda not available")
        device = torch.device("cpu")

    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config["model"]

    if args.experiment is not None:
        EXP_NAME = args.experiment
    else:
        EXP_NAME = config["experiment"]

    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config["dataset"]

    dataset = LoadData(DATASET_NAME)

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config["out_dir"]

    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.num_split is not None:
        params['num_split'] = int(args.num_split)
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
    net_params = config["net_params"]
    net_params["device"] = device
    net_params["gpu_id"] = config["gpu"]["id"]
    net_params["batch_size"] = params["batch_size"]

    """
    Model parameters
    """
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.graph_pool is not None:
        net_params['graph_pool'] = args.graph_pool
    if args.neighbor_pool is not None:
        net_params['neighbor_pool'] = args.neighbor_pool
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False
    if args.activation is not None:
        net_params['activation'] = args.activation
    if args.mlp_layers is not None:
        net_params['mlp_layers'] = int(args.mlp_layers)
    if args.bias is not None:
        net_params['bias'] = True if args.bias == 'True' else False
    if args.dgl_builtin is not None:
        net_params['dgl_builtin'] = True if args.dgl_builtin == 'True' else False

    """
    Expander parameters
    """
    if args.density is not None:
        net_params['density'] = float(args.density)
    if args.linear_type is not None:
        net_params['linear_type'] = args.linear_type
    if args.sampler is not None:
        net_params['sampler'] = args.sampler

    """
    MLP parameters
    """
    if args.gated is not None:
        net_params['gated'] = True if args.gated == 'True' else False

    """
    PNA parameters
    """
    if args.aggregators is not None:
        net_params['aggregators'] = args.aggregators
    if args.scalers is not None:
        net_params['scalers'] = args.scalers
    if args.num_tower is not None:
        net_params['num_tower'] = int(args.num_tower)
    if args.divide_input is not None:
        net_params['divide_input'] = True if args.divide_input == 'True' else False
    if args.gru is not None:
        net_params['gru'] = args.gru if args.gru == 'True' else False
    if args.edge_dim is not None:
        net_params['edge_dim'] = int(args.edge_dim)
    if args.num_pretrans_layer is not None:
        net_params['num_pretrans_layer'] = int(args.num_pretrans_layer)
    if args.num_posttrans_layer is not None:
        net_params['num_posttrans_layer'] = int(args.num_posttrans_layer)
    if args.use_simplified_version is not None:
        net_params['use_simplified_version'] = args.use_simplified_version \
            if args.use_simplified_version == 'True' else False

    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = len(np.unique(dataset.all.graph_labels))
    net_params['n_classes'] = num_classes

    def name_folder_path(x):
        return "{}{}{}_{}_{}_density_{}".format(out_dir, x, EXP_NAME, MODEL_NAME, DATASET_NAME, net_params["density"])

    root_log_dir = name_folder_path("logs/")
    root_ckpt_dir = name_folder_path("checkpoints/")
    write_file_name = name_folder_path("results/result_")
    write_config_file = name_folder_path("configs/config_")
    write_expander_dir = name_folder_path("expanders/")
    write_weight_dir = name_folder_path("expander_weights/")
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file, write_expander_dir, write_weight_dir

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)


if __name__ == '__main__':
    main()